import serial
import time
import serial.tools.list_ports
from typing import List, Optional, Tuple
import logging

# 配置日志系统
logging.basicConfig(
    level=logging.DEBUG,  # 改为DEBUG级别，查看更多细节
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def send_hex_data(ser: serial.Serial, hex_data: str) -> bool:
    try:
        data_bytes = bytes.fromhex(hex_data)
        bytes_sent = ser.write(data_bytes)
        logger.debug(f"发送请求帧: {hex_data}（{bytes_sent}字节）")
        return True
    except serial.SerialException as e:
        logger.error(f"串口发送错误: {e}")
        return False
    except ValueError as e:
        logger.error(f"16进制转换错误: {e}，数据: {hex_data}")
        return False
    except Exception as e:
        logger.error(f"发送未知错误: {e}")
        return False

def read_serial_response(ser: serial.Serial, timeout: float = 0.5) -> Optional[bytes]:
    """循环读取串口数据，确保长帧接收完整"""
    try:
        start_time = time.time()
        response = b""
        # 先清空接收缓存，避免历史数据干扰
        # ser.flushInput()
        time.sleep(0.1)
        
        # 循环读取，直到超时或数据停止接收
        while time.time() - start_time < timeout:
            if ser.in_waiting > 0:
                response += ser.read(ser.in_waiting)  # 读取所有可用数据
                start_time = time.time()  # 有新数据则重置超时
                time.sleep(0.005)  # 等待后续数据片段
            time.sleep(0.001)
        
        if response:
            logger.debug(f"接收完整应答帧（{len(response)}字节）: {response.hex()}")
            return response
        return None
    except serial.SerialException as e:
        logger.error(f"串口读取错误: {e}")
        return None
    except Exception as e:
        logger.error(f"读取数据异常: {e}")
        return None

def calculate_lrc(data: bytes) -> int:
    """计算LRC：累加所有字节→取反→加1→保留低8位"""
    lrc = 0
    for byte in data:
        lrc = (lrc + byte) & 0xFF  # 8位累加，防止溢出
    lrc = ((~lrc) + 1) & 0xFF     # 补码计算（取反加1）
    return lrc

def get_ser_response(ser: serial.Serial, command: str, data_length: int = 0) -> Optional[str]:
    command_config = {
        "get_data": {
            "fixed_fields": "03 00 FB 0E 04 00 00",
            "sleep": 0.2,  
            "parse": "hex"
        },
        "get_resultant_data": {
            "fixed_fields": "03 00 FB F0 03 00 00",
            "sleep": 0.2,  
            "parse": "force"
        }
    }

    if command not in command_config:
        logger.error(f"未知命令: {command}")
        return None
    config = command_config[command]

    # 协议固定参数
    frame_head_req = "55AA"
    frame_head_resp = "AA55"
    fixed_fields = config["fixed_fields"].replace(" ", "")
    sleep_time = config["sleep"]
    parse_type = config["parse"]

    if data_length <= 0:
        logger.error(f"{command}命令需指定有效数据长度（>0），当前: {data_length}")
        return None

    # 构建请求帧
    read_len_hex = data_length.to_bytes(2, byteorder='little').hex()
    data4_to_12 = f"{fixed_fields}{read_len_hex}"
    length_val = len(bytes.fromhex(data4_to_12))
    length_hex = length_val.to_bytes(2, byteorder='little').hex()
    req_frame_without_lrc = f"{frame_head_req}{length_hex}{data4_to_12}"

    # 计算LRC
    try:
        req_bytes_without_lrc = bytes.fromhex(req_frame_without_lrc)
        lrc_val = calculate_lrc(req_bytes_without_lrc)
        lrc_hex = f"{lrc_val:02X}"
    except ValueError as e:
        logger.error(f"请求帧LRC计算错误: {e}，数据: {req_frame_without_lrc}")
        return None

    full_req_frame = f"{req_frame_without_lrc}{lrc_hex}"
    logger.debug(f"程序发送的请求帧: {full_req_frame}")

    # 发送请求帧
    if not send_hex_data(ser, full_req_frame):
        logger.error("请求帧发送失败")
        return None

    # 接收响应（只读取一次，避免重复读取导致数据丢失）
    time.sleep(sleep_time)  # 等待设备处理并返回
    response = read_serial_response(ser, timeout=0.5)  # 超时设为0.5秒
    if not response:
        logger.warning("未收到应答帧（可能等待时间不足）")
        return None
    resp_len = len(response)

    # 补读可能的剩余数据（如果响应不完整）
    if resp_len < 18:  # 最小帧长度检查
        time.sleep(0.05)
        response += ser.read_all()
        resp_len = len(response)
        logger.debug(f"补读后应答帧（{resp_len}字节）: {response.hex()}")

    # 验证帧头
    if response[:2] != bytes.fromhex(frame_head_resp):
        logger.warning(f"应答帧头不匹配，期望{frame_head_resp}，实际{response[:2].hex()}")
        return None

    # 解析应答帧Length
    try:
        resp_length_val = int.from_bytes(response[2:4], byteorder='little')
        lrc_pos = 4 + resp_length_val
        if lrc_pos >= resp_len:
            logger.error(f"应答帧长度不足，LRC位置{lrc_pos}超出总长度{resp_len}")
            return None
    except IndexError as e:
        logger.error(f"应答帧Length解析错误: {e}")
        return None

    # 验证LRC
    resp_bytes_without_lrc = response[:lrc_pos]
    resp_lrc_calc = calculate_lrc(resp_bytes_without_lrc)
    resp_lrc_actual = response[lrc_pos]
    if resp_lrc_calc != resp_lrc_actual:
        logger.error(f"应答帧LRC校验失败！计算{resp_lrc_calc:02X}，实际{resp_lrc_actual:02X}")
        return None
    logger.debug("应答帧LRC校验通过")

    # 解析状态
    try:
        status = response[13]
        if status != 0x00:
            logger.warning(f"设备返回错误状态码: {status:02X}")
            return None
    except IndexError as e:
        logger.error(f"应答帧状态解析错误: {e}")
        return None

    # 解析返回字节数N
    try:
        return_len = int.from_bytes(response[11:13], byteorder='little')  
        if return_len <= 0:
            logger.warning("应答帧返回字节数为0")
            return None
    except IndexError as e:
        logger.error(f"应答帧返回字节数解析错误: {e}")
        return None

    # 提取数据域
    try:
        data_start = 14
        data_end = data_start + return_len
        if data_end > resp_len:
            logger.error(f"数据域超出应答帧长度，需{data_end}字节，实际{resp_len}字节")
            return None
        data_field = response[data_start:data_end]
    except IndexError as e:
        logger.error(f"数据域提取错误: {e}")
        return None

def main():
    logger.info("扫描可用串口...")
    available_ports = list(serial.tools.list_ports.comports())
    if not available_ports:
        logger.error("无可用串口")
        print("无可用串口，请检查硬件")
        return

    print("可用串口:")
    for port in available_ports:
        print(f"  {port.device} - {port.description}")

    try:
        # 配置串口参数（921600波特率）
        ser = serial.Serial(
            port=available_ports[0].device,
            baudrate=921600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1,
            write_timeout=0.5,
            inter_byte_timeout=0.001,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False
        )
        logger.info(f"连接串口{ser.name}成功（波特率921600）")
        print(f"\n连接{ser.name}成功，开始读取数据...")

        while True:
            # 读取合力数据
            get_ser_response(ser, "get_resultant_data", 3)
                
            time.sleep(1)

            # 读取测点数据
            get_ser_response(ser, "get_data", 32)

            time.sleep(1)

    except serial.SerialException as e:
        print(f"串口连接失败: {e}）")
        logger.error(f"串口连接错误: {e}")
    except KeyboardInterrupt:
        print("\n用户中断程序")
        logger.info("用户中断")
    except Exception as e:
        print(f"程序异常: {e}")
        logger.error(f"程序异常: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("串口已关闭")
            logger.info("串口关闭")

if __name__ == "__main__":
    main()
