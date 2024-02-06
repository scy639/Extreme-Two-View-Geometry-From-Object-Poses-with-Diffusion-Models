"""
递归遍历本文件夹下所有.py，并执行：
1. 若注释中包含中文，去除本注释
2. 若注释中包含scy(word match),去除scy这个词
"""
import os
import re


SKIP_FILE_or_FOLDER=['process_before_release.py','dataset_util.py']
# Function to remove Chinese comments from a string
def remove_chinese_comments(comment):
    return re.sub(r'#.*?[\u4e00-\u9fff]+.*?(?=\n|$)', '', comment)

# Function to remove the word "scy" from comments
def remove_scy_word(comment:str):
    if '#' not in comment:
        return comment
    #remove all '  ' after '#'
    i_comment=comment.index('#')
    s0=comment[:i_comment]
    s1=comment[i_comment:]
    s1=s1.replace(' scy ','  ')
    s1=s1.replace('# ','# ')
    s1=s1.replace(' scy:',' ')
    s1=s1.replace('#','#')
    s1=re.sub(' scy$',' ',s1)
    s1=re.sub('#scy$','',s1)
    return s0+s1

# Recursive function to process files in a directory
def process_files(directory):
    for filename in os.listdir(directory):
        if filename in SKIP_FILE_or_FOLDER:
            continue
        file_path = os.path.join(directory, filename)
        if os.path.isdir(file_path):
            process_files(file_path)
        elif filename.endswith('.py'):
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            modified_lines = []
            for line in lines:
                if line.strip().endswith('"""') or line.strip().endswith("'''"):
                    modified_line = line
                else:
                    if '#' in line:
                        modified_line = remove_scy_word(remove_chinese_comments(line))
                    else:
                        modified_line = line
                modified_lines.append(modified_line)

            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(modified_lines)
                

# Start processing files in the current directory
process_files(os.getcwd())




