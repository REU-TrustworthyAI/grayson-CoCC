import os
import re
from multiprocessing import Pool, cpu_count
from sentence_transformers import SentenceTransformer

# --- Regex Compilations (for efficiency) ---
OLD_COMMENT_REGEX = re.compile(r"oldComment:\n(.*?)\noldCode:", re.DOTALL)
OLD_CODE_REGEX = re.compile(r"oldCode:\n(.*?)\n\nnewComment:", re.DOTALL)
NEW_COMMENT_REGEX = re.compile(r"newComment:\n(.*?)\nnewCode:", re.DOTALL)
NEW_CODE_REGEX = re.compile(r"newCode:\n(.*?)\n\nstartline:", re.DOTALL)
LABEL_REGEX = re.compile(r"label:(\d+)")

# Regex for camel case splitting
CAMEL_CASE_SPLIT_REGEX = re.compile(r'(?<!^)(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')

def split_camel_case(text):
    """
    Splits a camelCase or PascalCase string into separate words.
    Example: "helloWorld" -> "hello World"
             "HTTPRequest" -> "HTTP Request"
    """
    if text is None:
        return None
    return ' '.join(CAMEL_CASE_SPLIT_REGEX.sub(' ', text).split())

def parse_code_change_file(file_content):
    """
    Parses the content of a code change file to extract old/new comments,
    old/new code, and the label. It also breaks up words based on camel case
    in the extracted code and comment sections.

    Args:
        file_content (str): The full content of the file as a string.

    Returns:
        dict: A dictionary containing 'oldComment', 'oldCode', 'newComment',
              'newCode', and 'label'. Returns None for missing sections.
              The code and comment strings will have camel case words split.
    """
    data = {
        "oldComment": None,
        "oldCode": None,
        "newComment": None,
        "newCode": None,
        "label": None,
    }

    if old_comment_match := OLD_COMMENT_REGEX.search(file_content):
        data["oldComment"] = split_camel_case(old_comment_match.group(1).strip())

    if old_code_match := OLD_CODE_REGEX.search(file_content):
        data["oldCode"] = split_camel_case(old_code_match.group(1).strip())

    if new_comment_match := NEW_COMMENT_REGEX.search(file_content):
        data["newComment"] = split_camel_case(new_comment_match.group(1).strip())

    if new_code_match := NEW_CODE_REGEX.search(file_content):
        data["newCode"] = split_camel_case(new_code_match.group(1).strip())

    if label_match := LABEL_REGEX.search(file_content):
        data["label"] = int(label_match.group(1))

    return data

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
        #_model.similarity_fn_name = "euclidean"

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
    # Use a flag to track if a particular text segment was added to `texts_to_encode`
    # and store its original presence for accurate indexing.
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
        old_code_old_comment_sim = 0

        # Dynamically find indices based on what was actually encoded
        # This approach ensures correct indexing even if some fields are missing.
        encoded_text_map = {text: idx for idx, text in enumerate(texts_to_encode)}

        if new_code_present and old_comment_present:
            new_code_idx = encoded_text_map[data["newCode"]]
            old_comment_idx = encoded_text_map[data["oldComment"]]
            new_code_old_comment_sim = similarity_matrix[new_code_idx, old_comment_idx].item()

        if old_code_present and old_comment_present:
            old_code_idx = encoded_text_map[data["oldCode"]]
            old_comment_idx = encoded_text_map[data["oldComment"]] # Re-use or get again
            old_code_old_comment_sim = similarity_matrix[old_code_idx, old_comment_idx].item()

        return {
            "label": data["label"],
            "new_code_old_comment_similarity": new_code_old_comment_sim,
            "old_code_old_comment_similarity": old_code_old_comment_sim
        }
    except Exception as e:
        print(f"Error during similarity computation for data with label {data.get('label')}: {e}")
        return None


def process_java_file(filepath):
    """
    Reads a Java file, parses its content, and computes similarity scores.
    Designed to be run in a multiprocessing pool.
    Returns the raw similarity results to be aggregated in the main process.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            file_content = f.read()
        data = parse_code_change_file(file_content)
        if data:
            return compute_similarity_for_process(data)
        else:
            print(f"Warning: Could not parse content from {filepath}")
            return None
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def traverse_folder_for_similarity(root_path):
    """
    Efficiently traverses a folder to find and process Java files for similarity.
    Uses os.walk for traversal and multiprocessing for parallel processing.
    Aggregates results after all files are processed, including the difference.
    """
    java_files = []
    for dirpath, _, filenames in os.walk(root_path):
        for f in filenames:
            if f.endswith('.java') and "122" in f:
                java_files.append(os.path.join(dirpath, f))

    num_processes = max(1, cpu_count() - 1)
    print(f"Processing {len(java_files)} Java files using {num_processes} processes...")

    all_results = []
    if java_files:
        with Pool(processes=num_processes, initializer=_initializer) as pool:
            all_results = pool.map(process_java_file, java_files)

    numProcessed_0 = 0
    numProcessed_1 = 0
    total_new_code_old_comment_similarity_0 = 0
    total_new_code_old_comment_similarity_1 = 0
    total_old_code_old_comment_similarity_0 = 0 # Track for label 0
    total_old_code_old_comment_similarity_1 = 0 # Track for label 1
    total_processed_files = 0 # This one is just for the overall average of old code to old comment

    # --- New variables for difference calculation ---
    total_difference_label_0 = 0
    total_difference_label_1 = 0
    # -----------------------------------------------

    for result in all_results:
        if result is not None:
            # We will calculate total_processed_files and the overall avgOldCodeOldCommentSimilarity
            # after iterating through all results for more accuracy.

            # Accumulate totals for different labels
            if result["label"] == 0:
                total_new_code_old_comment_similarity_0 += result["new_code_old_comment_similarity"]
                total_old_code_old_comment_similarity_0 += result["old_code_old_comment_similarity"]
                numProcessed_0 += 1
                total_difference_label_0 += (result["new_code_old_comment_similarity"] - result["old_code_old_comment_similarity"])
            elif result["label"] == 1:
                total_new_code_old_comment_similarity_1 += result["new_code_old_comment_similarity"]
                total_old_code_old_comment_similarity_1 += result["old_code_old_comment_similarity"]
                numProcessed_1 += 1
                total_difference_label_1 += (result["new_code_old_comment_similarity"] - result["old_code_old_comment_similarity"])

    avgNewCodeOldCommentSimilarity_0 = (total_new_code_old_comment_similarity_0 / numProcessed_0) if numProcessed_0 > 0 else 0
    avgNewCodeOldCommentSimilarity_1 = (total_new_code_old_comment_similarity_1 / numProcessed_1) if numProcessed_1 > 0 else 0

    # Calculate average old code to old comment similarity separately for each label
    avgOldCodeOldCommentSimilarity_0 = (total_old_code_old_comment_similarity_0 / numProcessed_0) if numProcessed_0 > 0 else 0
    avgOldCodeOldCommentSimilarity_1 = (total_old_code_old_comment_similarity_1 / numProcessed_1) if numProcessed_1 > 0 else 0

    # Calculate overall average for Old Code to Old Comment for combined context, if desired
    # Or, you can just show per-label averages.
    # For now, let's keep it per label for a clearer comparison of differences.

    avg_difference_label_0 = (total_difference_label_0 / numProcessed_0) if numProcessed_0 > 0 else 0
    avg_difference_label_1 = (total_difference_label_1 / numProcessed_1) if numProcessed_1 > 0 else 0

    print(f"\nAggregated Similarity Results:")
    print(f"--- For Label 0 (e.g., negative; should be high) ---")
    print(f"  Avg New Code to Old Comment (Label 0): {avgNewCodeOldCommentSimilarity_0:.4f}")
    print(f"  Avg Old Code to Old Comment (Label 0): {avgOldCodeOldCommentSimilarity_0:.4f}")
    print(f"  Avg Difference (New Code-Old Comment - Old Code-Old Comment) for Label 0: {avg_difference_label_0:.4f}")

    print(f"\n--- For Label 1 (e.g., positive; should be low) ---")
    print(f"  Avg New Code to Old Comment (Label 1): {avgNewCodeOldCommentSimilarity_1:.4f}")
    print(f"  Avg Old Code to Old Comment (Label 1): {avgOldCodeOldCommentSimilarity_1:.4f}")
    print(f"  Avg Difference (New Code-Old Comment - Old Code-Old Comment) for Label 1: {avg_difference_label_1:.4f}")


# Example Usage (for testing purposes, creates dummy files and then cleans up)
if __name__ == "__main__":
    traverse_folder_for_similarity("features")