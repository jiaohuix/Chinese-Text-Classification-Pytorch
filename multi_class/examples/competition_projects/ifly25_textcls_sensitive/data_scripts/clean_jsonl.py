import json
import sys

def clean_jsonl(input_filepath, output_filepath):
    """
    读取一个可能存在格式问题的JSONL文件，
    然后输出一个没有问题的JSONL文件。

    Args:
        input_filepath (str): 输入的JSONL文件路径。
        output_filepath (str): 输出的无格式问题JSONL文件路径。
    """
    valid_lines = []
    error_lines = []
    line_num = 0

    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            for line in infile:
                line_num += 1
                # 移除行首尾的空白字符，特别是可能存在的额外换行符
                stripped_line = line.strip()
                if not stripped_line: # 跳过空行
                    continue

                try:
                    # 尝试解析每一行作为一个JSON对象
                    json_object = json.loads(stripped_line)
                    # 确保解析出来的确实是一个对象（字典），而不是列表或其他类型（虽然JSONL通常是对象）
                    if isinstance(json_object, dict):
                        valid_lines.append(json_object)
                    else:
                        # 如果解析成功但不是预期的字典类型，也视为一种“问题”
                        error_lines.append((line_num, f"Parsed but not a JSON object (dictionary): {stripped_line}"))
                except json.JSONDecodeError as e:
                    # 捕获JSON解析错误
                    error_lines.append((line_num, f"JSONDecodeError: {e} - Content: {stripped_line}"))
                except Exception as e:
                    # 捕获其他可能的异常
                    error_lines.append((line_num, f"Unexpected error: {e} - Content: {stripped_line}"))

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_filepath}'", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error reading input file '{input_filepath}': {e}", file=sys.stderr)
        return

    # 将有效的JSON对象写入输出文件
    try:
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            for obj in valid_lines:
                # 使用ensure_ascii=False以正确处理非ASCII字符，并设置indent=None（默认值）以保证JSONL格式
                json_line = json.dumps(obj, ensure_ascii=False, indent=None)
                outfile.write(json_line + '\n')
        print(f"Successfully processed '{input_filepath}'.")
        print(f"Valid JSONL written to '{output_filepath}'.")
    except Exception as e:
        print(f"Error writing output file '{output_filepath}': {e}", file=sys.stderr)

    # 打印出错的行信息
    if error_lines:
        print("\n--- Errors Encountered ---")
        for num, error_msg in error_lines:
            print(f"Line {num}: {error_msg}", file=sys.stderr)
        print(f"\n{len(error_lines)} lines encountered errors and were skipped.")
    else:
        print("No errors encountered during processing.")

# --- 如何使用 ---
if __name__ == "__main__":
    # 创建一个示例的、可能包含错误格式的JSONL文件
  
    # input_filename = "newtest.jsonl"
    # output_filename = "output_clean.jsonl"
    import sys
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    try:
      
        # 调用函数进行处理
        clean_jsonl(input_filename, output_filename)

    except Exception as e:
        print(f"An error occurred during sample file creation or processing: {e}", file=sys.stderr)

