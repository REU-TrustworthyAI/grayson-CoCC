import os
import re
from multiprocessing import Pool, cpu_count
from sentence_transformers import SentenceTransformer

# --- Regex Compilations (for efficiency) ---
# Java-specific regexes (kept for context)
OLD_COMMENT_REGEX = re.compile(r"oldComment:\n(.*?)\noldCode:", re.DOTALL)
OLD_CODE_REGEX = re.compile(r"oldCode:\n(.*?)\n\nnewComment:", re.DOTALL)
NEW_COMMENT_REGEX = re.compile(r"newComment:\n(.*?)\nnewCode:", re.DOTALL)
NEW_CODE_REGEX = re.compile(r"newCode:\n(.*?)\n\nstartline:", re.DOTALL)
LABEL_REGEX = re.compile(r"label:(\d+)")

# Regex for camel case splitting
CAMEL_CASE_SPLIT_REGEX = re.compile(r'(?<!^)(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')

# Regex to remove non-alphanumeric characters (except spaces after splitting)
NON_ALPHANUMERIC_REGEX = re.compile(r'[^a-zA-Z0-9\s]')

# --- NEW Regex Patterns for Python Delimiters ---
# Matches a line consisting of one or more hyphens
PYTHON_OLD_NEW_DELIMITER_REGEX = re.compile(r'^\s*-+$', re.MULTILINE)
# Matches a line consisting of one or more equals signs
PYTHON_END_DELIMITER_REGEX = re.compile(r'^\s*=+$', re.MULTILINE)

def split_camel_case_and_clean(text):
    """
    Splits camelCase/PascalCase, removes non-alphanumeric characters,
    standardizes spaces, and lowercases the text.
    """
    if text is None:
        return None

    text = CAMEL_CASE_SPLIT_REGEX.sub(' ', text)
    #text = NON_ALPHANUMERIC_REGEX.sub(' ', text)
    #text = text.lower()
    text = ' '.join(text.split()).strip()

    return text

def extract_python_comments(code_block_text):
    """
    Extracts and cleans single-line comments (#) from a Python code block.
    """
    comments = []
    lines = code_block_text.splitlines()
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('#'):
            comment_text = split_camel_case_and_clean(stripped_line[1:].strip())
            if comment_text:
                comments.append(comment_text)
    return comments

def parse_code_change_file(file_content):
    """
    Parses the content of a code change file (Java format).
    """
    data = {
        "oldComment": None,
        "oldCode": None,
        "newComment": None,
        "newCode": None,
        "label": None,
    }

    if old_comment_match := OLD_COMMENT_REGEX.search(file_content):
        data["oldComment"] = split_camel_case_and_clean(old_comment_match.group(1).strip())

    if old_code_match := OLD_CODE_REGEX.search(file_content):
        data["oldCode"] = split_camel_case_and_clean(old_code_match.group(1).strip())

    if new_comment_match := NEW_COMMENT_REGEX.search(file_content):
        data["newComment"] = split_camel_case_and_clean(new_comment_match.group(1).strip())

    if new_code_match := NEW_CODE_REGEX.search(file_content):
        data["newCode"] = split_camel_case_and_clean(new_code_match.group(1).strip())

    if label_match := LABEL_REGEX.search(file_content):
        data["label"] = int(label_match.group(1))

    return data

def parse_python_code_change(file_content):
    """
    Parses content from the specific Python code change format,
    handling variable-length delimiters.
    """
    # Split by the first delimiter (line of hyphens)
    parts_by_hyphens = PYTHON_OLD_NEW_DELIMITER_REGEX.split(file_content)

    # We expect 2 parts: [content before ---, content after ---]
    if len(parts_by_hyphens) < 2:
        return None # First delimiter not found in the expected format

    old_code_block_raw = parts_by_hyphens[0].strip()

    # The second part contains the new code block and the final delimiter (line of equals)
    new_code_and_rest = parts_by_hyphens[1]

    # Split the remainder by the second delimiter (line of equals signs)
    parts_by_equals = PYTHON_END_DELIMITER_REGEX.split(new_code_and_rest)

    # We expect at least one part (the new code block itself)
    if len(parts_by_equals) < 1:
        return None # Second delimiter not found or not in expected format

    new_code_block_raw = parts_by_equals[0].strip()

    # If either code block is empty after stripping, it's likely a malformed file
    if not old_code_block_raw or not new_code_block_raw:
        return None

    # Extract comments from raw code blocks and clean them
    old_comments_list = extract_python_comments(old_code_block_raw)
    new_comments_list = extract_python_comments(new_code_block_raw)

    # Determine label based on comment change
    label = 1 if old_comments_list != new_comments_list else 0

    # Clean entire code blocks for similarity comparison
    cleaned_old_code = split_camel_case_and_clean(old_code_block_raw)
    cleaned_new_code = split_camel_case_and_clean(new_code_block_raw)

    return {
        "oldCode": cleaned_old_code,
        "newCode": cleaned_new_code,
        "oldComment": " ".join(old_comments_list), # Join cleaned comments into a single string
        "newComment": " ".join(new_comments_list), # Join cleaned comments into a single string
        "label": label,
    }

# Global variable to store the model, will be loaded once per process
_model = None

def _initializer():
    """
    Initializer function for the multiprocessing pool.
    Loads the SentenceTransformer model once per child process.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_similarity_for_process(data):
    """
    Computes similarity scores for a single parsed data dictionary.
    This function uses the global _model loaded by the initializer.
    It returns the individual similarity scores and the label,
    which will be aggregated in the main process.
    """
    global _model
    if _model is None:
        _initializer()

    texts_to_encode = []
    new_code_present = False
    old_comment_present = False
    old_code_present = False

    if data["newCode"] is not None and data["newCode"].strip():
        texts_to_encode.append(data["newCode"])
        new_code_present = True
    if data["oldComment"] is not None and data["oldComment"].strip():
        texts_to_encode.append(data["oldComment"])
        old_comment_present = True
    if data["oldCode"] is not None and data["oldCode"].strip():
        texts_to_encode.append(data["oldCode"])
        old_code_present = True

    if len(texts_to_encode) < 2:
        print(f"Warning: Not enough valid text segments to compute similarity for data with label {data.get('label')}.")
        return None

    try:
        embeddings = _model.encode(texts_to_encode, normalize_embeddings=True)
        similarity_matrix = _model.similarity(embeddings, embeddings)

        new_code_old_comment_sim = 0
        old_code_new_code_sim = 0
        old_code_old_comment_sim = 0

        encoded_text_map = {text: idx for idx, text in enumerate(texts_to_encode)}

        if new_code_present and old_comment_present:
            new_code_idx = encoded_text_map[data["newCode"]]
            old_comment_idx = encoded_text_map[data["oldComment"]]
            new_code_old_comment_sim = similarity_matrix[new_code_idx, old_comment_idx].item()

        if old_code_present and old_comment_present:
            old_code_idx = encoded_text_map[data["oldCode"]]
            old_comment_idx = encoded_text_map[data["oldComment"]]
            old_code_old_comment_sim = similarity_matrix[old_code_idx, old_comment_idx].item()
            new_code_old_comment_sim = similarity_matrix[new_code_idx, old_comment_idx].item()

        if old_code_present and new_code_present:
            old_code_idx = encoded_text_map[data["oldCode"]]
            new_code_idx = encoded_text_map[data["newCode"]]
            old_code_new_code_sim = similarity_matrix[old_code_idx, new_code_idx].item()

        return {
            "label": data["label"],
            "new_code_old_comment_similarity": new_code_old_comment_sim,
            "old_code_new_code_similarity": old_code_new_code_sim,
            "old_code_new_comment_similarity": new_code_old_comment_sim,
            "old_code_old_comment_similarity": old_code_old_comment_sim
        }
    except Exception as e:
        print(f"Error during similarity computation for data with label {data.get('label')}: {e}")
        return None


def process_file_for_similarity(filepath, parser_func):
    """
    Reads a file, parses its content using the provided parser_func,
    and computes similarity scores.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            file_content = f.read()
        data = parser_func(file_content)
        if data:
            return compute_similarity_for_process(data)
        else:
            print(f"Warning: Could not parse content from {filepath} with {parser_func.__name__}")
            return None
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def traverse_folder_for_similarity(root_path):
    """
    Efficiently traverses a folder to find and process Java and Python files for similarity.
    Dispatches to the correct parsing function based on file content/extension.
    """
    files_to_process = []
    for dirpath, _, filenames in os.walk(root_path):
        for f in filenames:
            file_path = os.path.join(dirpath, f)
            # Determine which parser to use based on file extension and potential content
            if f.endswith('.py'):
                # For Python files, assume the new Python format
                files_to_process.append({'path': file_path, 'parser': parse_python_code_change})
            elif f.endswith('.java') and '122' in f:
                # For Java files, assume the existing Java format
                files_to_process.append({'path': file_path, 'parser': parse_code_change_file})


    num_processes = max(1, cpu_count() - 1)
    print(f"Processing {len(files_to_process)} files using {num_processes} processes...")

    all_results = []
    if files_to_process:
        with Pool(processes=num_processes, initializer=_initializer) as pool:
            # Prepare arguments for map: (filepath, parser_func)
            # Use starmap to pass multiple arguments to `process_file_for_similarity`
            tasks = [(info['path'], info['parser']) for info in files_to_process]
            all_results = pool.starmap(process_file_for_similarity, tasks)


    numProcessed_0 = 0
    numProcessed_1 = 0
    total_new_code_old_comment_similarity_0 = 0
    total_new_code_old_comment_similarity_1 = 0
    total_old_code_old_comment_similarity_0 = 0
    total_old_code_old_comment_similarity_1 = 0
    total_old_code_new_code_similarity_0 = 0
    total_old_code_new_code_similarity_1 = 0
    total_difference_label_0 = 0
    total_difference_label_1 = 0

    for result in all_results:
        if result is not None:
            if result["label"] == 0:
                total_new_code_old_comment_similarity_0 += result["new_code_old_comment_similarity"]
                total_old_code_old_comment_similarity_0 += result["old_code_old_comment_similarity"]
                total_old_code_new_code_similarity_0 += result["old_code_new_code_similarity"]
                numProcessed_0 += 1
                total_difference_label_0 += (result["new_code_old_comment_similarity"] - result["old_code_old_comment_similarity"])
            elif result["label"] == 1:
                total_new_code_old_comment_similarity_1 += result["new_code_old_comment_similarity"]
                total_old_code_old_comment_similarity_1 += result["old_code_old_comment_similarity"]
                total_old_code_new_code_similarity_1 += result["old_code_new_code_similarity"]
                numProcessed_1 += 1
                total_difference_label_1 += (result["new_code_old_comment_similarity"] - result["old_code_old_comment_similarity"])

    avgNewCodeOldCommentSimilarity_0 = (total_new_code_old_comment_similarity_0 / numProcessed_0) if numProcessed_0 > 0 else 0
    avgNewCodeOldCommentSimilarity_1 = (total_new_code_old_comment_similarity_1 / numProcessed_1) if numProcessed_1 > 0 else 0

    avgOldCodeOldCommentSimilarity_0 = (total_old_code_old_comment_similarity_0 / numProcessed_0) if numProcessed_0 > 0 else 0
    avgOldCodeOldCommentSimilarity_1 = (total_old_code_old_comment_similarity_1 / numProcessed_1) if numProcessed_1 > 0 else 0
    avgNewCodeOldCommentSimilarity_1 = (total_new_code_old_comment_similarity_1 / numProcessed_1) if numProcessed_1 > 0 else 0

    avgOldCodeNewCodeSimilarity_0 = (total_old_code_new_code_similarity_0 / numProcessed_0) if numProcessed_0 > 0 else 0
    avgOldCodeNewCodeSimilarity_1 = (total_old_code_new_code_similarity_1 / numProcessed_1) if numProcessed_1 > 0 else 0

    avg_difference_label_0 = (total_difference_label_0 / numProcessed_0) if numProcessed_0 > 0 else 0
    avg_difference_label_1 = (total_difference_label_1 / numProcessed_1) if numProcessed_1 > 0 else 0

    print(f"\nAggregated Similarity Results:")
    print(f"--- For Label 0 (e.g., negative; should be high) ---")
    print(f"  Avg New Code to Old Comment (Label 0): {avgNewCodeOldCommentSimilarity_0:.4f}")
    print(f"  Avg Old Code to Old Comment (Label 0): {avgOldCodeOldCommentSimilarity_0:.4f}")
    print(f"  Avg Old Code to New Code (Label 0): {avgOldCodeNewCodeSimilarity_0:.4f}")
    print(f"  Avg Difference (New Code-Old Comment - Old Code-Old Comment) for Label 0: {avg_difference_label_0:.4f}")

    print(f"\n--- For Label 1 (e.g., positive; should be low) ---")
    print(f"  Avg New Code to Old Comment (Label 1): {avgNewCodeOldCommentSimilarity_1:.4f}")
    print(f"  Avg Old Code to Old Comment (Label 1): {avgOldCodeOldCommentSimilarity_1:.4f}")
    print(f"  Avg Old Code to New Code (Label 1): {avgOldCodeNewCodeSimilarity_1:.4f}")
    print(f"  Avg Difference (New Code-Old Comment - Old Code-Old Comment) for Label 1: {avg_difference_label_1:.4f}")


if __name__ == "__main__":
    traverse_folder_for_similarity("python_ccset_raw")