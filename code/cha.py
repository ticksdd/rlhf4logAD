import json
import re

def replace_conversation_value(data, new_instruction):
    """
    替换JSON数据中conversations字段的value值
    保留原始日志序列，仅替换指令部分
    
    参数:
        data: 原始JSON数据（列表）
        new_instruction: 新的指令文本（字符串）
    
    返回:
        替换后的JSON数据
    """
    # 正则表达式匹配日志序列部分
    log_pattern = re.compile(r"(the given log entry is as follows: )(\[.*?\])", re.DOTALL)
    
    for item in data:
        conversations = item.get("conversations", [])
        for conv in conversations:
            if conv.get("from") == "human":
                original_value = conv.get("value", "")
                
                # 提取日志序列部分
                match = log_pattern.search(original_value)
                if match:
                    log_section = match.group(2)
                    # 拼接新指令和日志序列
                    new_value = f"{new_instruction}\n{log_section}"
                    conv["value"] = new_value
                    
    return data

# 示例用法
if __name__ == "__main__":
    # 读取原始JSON文件
    with open("./dpo.json", "r") as f:
        data = json.load(f)
    
    # 定义新指令（需符合用户提供的指令模板）
    new_instruction = "You are a log analysis expert tasked with determining whether any individual log in the given sequence is abnormal (e.g., error, interrupt, crashes, security violations). Evaluate each log independently for abnormalities, Ignoring sequence-level patterns. As long as one log is considered abnormal, the final judgment result is abnormal. If all logs are normal, the final judgment result is normal.  Don't be too sensitive, no obvious errors will be considered normal(e.g alignment exceptions, generating core, are considered as normal), and corrected errors will also be considered normal). Output Requirements: Firstly, anomaly status (Strictly limit the format to [anomaly status: #Abnormal# / #Normal#]).  Analysis of Log sequence: a. Given a brief summery of this log sequnce. b.  Identify triggered security protocol rules and anomaly patterns. c. Analyze and provide the anomaly severity score for this log sequence.(1-10). "
    
    # 执行替换
    updated_data = replace_conversation_value(data, new_instruction)
    
    # 保存更新后的JSON
    with open("output.json", "w") as f:
        json.dump(updated_data, f, indent=2)