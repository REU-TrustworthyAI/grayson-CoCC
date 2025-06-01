import os
from sentence_transformers import SentenceTransformer
import re

def parse_code_change_file(file_content):
    """
    Parses the content of a code change file to extract old/new comments,
    old/new code, and the label.

    Args:
        file_content (str): The full content of the file as a string.

    Returns:
        dict: A dictionary containing 'oldComment', 'oldCode', 'newComment',
              'newCode', and 'label'. Returns None for missing sections.
    """

    data = {
        "oldComment": None,
        "oldCode": None,
        "newComment": None,
        "newCode": None,
        "label": None,
    }

    # Regex to find comments and code blocks
    # Using re.DOTALL to allow . to match newlines
    old_comment_match = re.search(r"oldComment:\n(.*?)\noldCode:", file_content, re.DOTALL)
    if old_comment_match:
        data["oldComment"] = old_comment_match.group(1).strip()

    old_code_match = re.search(r"oldCode:\n(.*?)\n\nnewComment:", file_content, re.DOTALL)
    if old_code_match:
        data["oldCode"] = old_code_match.group(1).strip()

    new_comment_match = re.search(r"newComment:\n(.*?)\nnewCode:", file_content, re.DOTALL)
    if new_comment_match:
        data["newComment"] = new_comment_match.group(1).strip()

    new_code_match = re.search(r"newCode:\n(.*?)\n\nstartline:", file_content, re.DOTALL)
    if new_code_match:
        data["newCode"] = new_code_match.group(1).strip()

    # Extract label
    label_match = re.search(r"label:(\d+)", file_content)
    if label_match:
        data["label"] = int(label_match.group(1))

    return data

model = SentenceTransformer("all-MiniLM-L6-v2")

numProcessed_0 = 0
numProcessed_1 = 0
avgNewCodeOldCommentSimilarity_0 = 0
avgNewCodeOldCommentSimilarity_1 = 0
avgOldCodeOldCommentSimilarity = 0

def compute_similarity(data):
    global avgNewCodeOldCommentSimilarity_0
    global avgNewCodeOldCommentSimilarity_1
    global avgOldCodeOldCommentSimilarity
    global numProcessed_0
    global numProcessed_1
    
    embeddings = model.encode([data["newCode"], data["oldComment"], data["oldCode"]], normalize_embeddings=True)
    similarity = model.similarity(embeddings,embeddings)
    avgOldCodeOldCommentSimilarity = (avgOldCodeOldCommentSimilarity * (numProcessed_0 + numProcessed_1) + similarity[1,2]) / (numProcessed_0 + numProcessed_1 + 1) 
    if data["label"] == 0:
        avgNewCodeOldCommentSimilarity_0 = (avgNewCodeOldCommentSimilarity_0 * numProcessed_0 + similarity[0,1]) / (numProcessed_0 + 1)
        numProcessed_0 += 1
    if data["label"] == 1:
        avgNewCodeOldCommentSimilarity_1 = (avgNewCodeOldCommentSimilarity_1 * numProcessed_1 + similarity[0,1]) / (numProcessed_1 + 1)
        numProcessed_1 += 1

def get_change_info(path):
    is_method = False
    with open(path) as f:
        change_line = []
        for line in f.readlines():
            if line.startswith('type:') and line.__contains__('METHOD_COMMENT'):
                is_method = True
            if line.startswith('changeNum:'):
                change_num = int(line.split(':')[-1])
            if line.startswith('label:'):
                label = int(line.split(':')[-1])
            if line.startswith('change ') and line.__contains__(':') and line.__contains__(','):
                sl, el = [int(x.replace(' ', '').replace('\n', '')) for x in line.split(':')[-1].split(',')]
                while sl <= el:
                    if sl not in change_line:
                        change_line.append(sl)
                    sl += 1
    if is_method:
        return 0, 0, 0
    return change_num, len(change_line), label


change_number = 0
changed_cnt = 0
unchanged_cnt = 0
method_cnt = 0


def traverse_folder_for_Q13(path):
    if os.path.isdir(path):
        for f in os.listdir(path):
            traverse_folder_for_Q13(os.path.join(path, f))
    else:
        if path.endswith('.java'):
            global change_number
            global unchanged_cnt
            global changed_cnt
            global method_cnt
            change_num, st_num, label = get_change_info(path)
            if change_num == 0 and st_num == 0 and label == 0:
                method_cnt += 1
            change_number += change_num
            if label == 1:
                changed_cnt += 1
            else:
                unchanged_cnt += 1

def traverse_folder_for_similarity(path):
    if os.path.isdir(path):
        for f in os.listdir(path):
            traverse_folder_for_similarity(os.path.join(path, f))
    else:
        if path.endswith('.java'):
            file_content = ""
            with open(path) as f:
                for line in f.readlines():
                    file_content += line
            data = parse_code_change_file(file_content)
            compute_similarity(data)

traverse_folder_for_similarity("features")
print(avgNewCodeOldCommentSimilarity_0)
print(avgNewCodeOldCommentSimilarity_1)
print(avgOldCodeOldCommentSimilarity)

# print(change_number)
# print(changed_cnt / (changed_cnt + unchanged_cnt) * 100)
# print(unchanged_cnt / (changed_cnt + unchanged_cnt) * 100)

cnt = 0
change_num_1_3_label_1 = 0
change_num_1_3_label_0 = 0

change_num_4_6_label_1 = 0
change_num_4_6_label_0 = 0

change_num_7_9_label_1 = 0
change_num_7_9_label_0 = 0

change_num_10_12_label_1 = 0
change_num_10_12_label_0 = 0

change_num_13_15_label_1 = 0
change_num_13_15_label_0 = 0

change_num_over_15_label_1 = 0
change_num_over_15_label_0 = 0

change_st_1_3_label_1 = 0
change_st_1_3_label_0 = 0

change_st_4_6_label_1 = 0
change_st_4_6_label_0 = 0

change_st_7_9_label_1 = 0
change_st_7_9_label_0 = 0

change_st_10_12_label_1 = 0
change_st_10_12_label_0 = 0

change_st_13_15_label_1 = 0
change_st_13_15_label_0 = 0

change_st_over_15_label_1 = 0
change_st_over_15_label_0 = 0


def traverse_folder_for_Q8(path):
    if os.path.isdir(path):
        for f in os.listdir(path):
            traverse_folder_for_Q8(os.path.join(path, f))
    else:
        if path.endswith('.java'):
            global cnt
            cnt += 1
            change_num, st_num, label = get_change_info(path)
            if label == 0:
                if st_num >= 1 and st_num <= 3:
                    global change_st_1_3_label_0
                    change_st_1_3_label_0 += 1
                if st_num >= 4 and st_num <= 6:
                    global change_st_4_6_label_0
                    change_st_4_6_label_0 += 1
                if st_num >= 7 and st_num <= 9:
                    global change_st_7_9_label_0
                    change_st_7_9_label_0 += 1
                if st_num >= 10 and st_num <= 12:
                    global change_st_10_12_label_0
                    change_st_10_12_label_0 += 1
                if st_num >= 13 and st_num <= 15:
                    global change_st_13_15_label_0
                    change_st_13_15_label_0 += 1
                if st_num > 15:
                    global change_st_over_15_label_0
                    change_st_over_15_label_0 += 1

                if change_num >= 1 and change_num <= 3:
                    global change_num_1_3_label_0
                    change_num_1_3_label_0 += 1
                if change_num >= 4 and change_num <= 6:
                    global change_num_4_6_label_0
                    change_num_4_6_label_0 += 1
                if change_num >= 7 and change_num <= 9:
                    global change_num_7_9_label_0
                    change_num_7_9_label_0 += 1
                if change_num >= 10 and change_num <= 12:
                    global change_num_10_12_label_0
                    change_num_10_12_label_0 += 1
                if change_num >= 13 and change_num <= 15:
                    global change_num_13_15_label_0
                    change_num_13_15_label_0 += 1
                if change_num > 15:
                    global change_num_over_15_label_0
                    change_num_over_15_label_0 += 1

            else:
                if st_num >= 1 and st_num <= 3:
                    global change_st_1_3_label_1
                    change_st_1_3_label_1 += 1
                if st_num >= 4 and st_num <= 6:
                    global change_st_4_6_label_1
                    change_st_4_6_label_1 += 1
                if st_num >= 7 and st_num <= 9:
                    global change_st_7_9_label_1
                    change_st_7_9_label_1 += 1
                if st_num >= 10 and st_num <= 12:
                    global change_st_10_12_label_1
                    change_st_10_12_label_1 += 1
                if st_num >= 13 and st_num <= 15:
                    global change_st_13_15_label_1
                    change_st_13_15_label_1 += 1
                if st_num > 15:
                    global change_st_over_15_label_1
                    change_st_over_15_label_1 += 1

                if change_num >= 1 and change_num <= 3:
                    global change_num_1_3_label_1
                    change_num_1_3_label_1 += 1
                if change_num >= 4 and change_num <= 6:
                    global change_num_4_6_label_1
                    change_num_4_6_label_1 += 1
                if change_num >= 7 and change_num <= 9:
                    global change_num_7_9_label_1
                    change_num_7_9_label_1 += 1
                if change_num >= 10 and change_num <= 12:
                    global change_num_10_12_label_1
                    change_num_10_12_label_1 += 1
                if change_num >= 13 and change_num <= 15:
                    global change_num_13_15_label_1
                    change_num_13_15_label_1 += 1
                if change_num > 15:
                    global change_num_over_15_label_1
                    change_num_over_15_label_1 += 1
                pass

# traverse_folder_for_Q8("/Users/chenyn/chenyn's/研究生/DataSet/My dect/data/回复/")
# print('total:', cnt)
# print('---------------------------')
# print(change_num_1_3_label_1)
# print(change_num_1_3_label_0)
# print(change_num_1_3_label_1 / (change_num_1_3_label_1 + change_num_1_3_label_0))
#
# print(change_num_4_6_label_1)
# print(change_num_4_6_label_0)
# print(change_num_4_6_label_1 / (change_num_4_6_label_1 + change_num_4_6_label_0))
#
# print(change_num_7_9_label_1)
# print(change_num_7_9_label_0)
# print(change_num_7_9_label_1 / (change_num_7_9_label_1 + change_num_7_9_label_0))
#
# print(change_num_10_12_label_1)
# print(change_num_10_12_label_0)
# print(change_num_10_12_label_1 / (change_num_10_12_label_1 + change_num_10_12_label_0))
#
# print(change_num_13_15_label_1)
# print(change_num_13_15_label_0)
# print(change_num_13_15_label_1 / (change_num_13_15_label_1 + change_num_13_15_label_0))
#
# print(change_num_over_15_label_1)
# print(change_num_over_15_label_0)
# print(change_num_over_15_label_1 / (change_num_over_15_label_1 + change_num_over_15_label_0))
# print('-----------------------------------')
# print(change_st_1_3_label_1)
# print(change_st_1_3_label_0)
# print(change_st_1_3_label_1 / (change_st_1_3_label_1 + change_st_1_3_label_0))
#
# print(change_st_4_6_label_1)
# print(change_st_4_6_label_0)
# print(change_st_4_6_label_1 / (change_st_4_6_label_1 + change_st_4_6_label_0))
#
# print(change_st_7_9_label_1)
# print(change_st_7_9_label_0)
# print(change_st_7_9_label_1 / (change_st_7_9_label_1 + change_st_7_9_label_0))
#
# print(change_st_10_12_label_1)
# print(change_st_10_12_label_0)
# print(change_st_10_12_label_1 / (change_st_10_12_label_1 + change_st_10_12_label_0))
#
# print(change_st_13_15_label_1)
# print(change_st_13_15_label_0)
# print(change_st_13_15_label_1 / (change_st_13_15_label_1 + change_st_13_15_label_0))
#
# print(change_st_over_15_label_1)
# print(change_st_over_15_label_0)
# print(change_st_over_15_label_1 / (change_st_over_15_label_1 + change_st_over_15_label_0))
