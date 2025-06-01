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

# Example usage with your provided content:
file_content = """
==========================================CCSet==========================================
oldComment:
/**
     * Enforce max field sizes according to SQL column definition.
     * SQL92 13.8
     */

oldCode:
    void enforceFieldValueLimits(Object[] row) throws HsqlException {

        int colindex;

        if (sqlEnforceSize || sqlEnforceStrictSize) {
            for (colindex = 0; colindex < iVisibleColumns; colindex++) {
                if (colSizes[colindex] != 0 && row[colindex] != null) {
                    row[colindex] = enforceSize(row[colindex],
                                                colTypes[colindex],
                                                colSizes[colindex], true,
                                                sqlEnforceStrictSize);
                }
            }
        }
    }


newComment:
/**
     * Enforce max field sizes according to SQL column definition.
     * SQL92 13.8
     */

newCode:
    void enforceFieldValueLimits(Object[] row) throws HsqlException {

        int colindex;

        if (sqlEnforceSize || sqlEnforceStrictSize) {
            for (colindex = 0; colindex < visibleColumnCount; colindex++) {
                if (colSizes[colindex] != 0 && row[colindex] != null) {
                    row[colindex] = enforceSize(row[colindex],
                                                colTypes[colindex],
                                                colSizes[colindex], true,
                                                sqlEnforceStrictSize);
                }
            }
        }
    }


startline:1849

endline:1867

type:METHOD_COMMENT

path:/Users/chenyn/chenyn's/研究生/DataSet/CommitData/论文/hsqldb/1725/old/src/Table.java

label:0

...
"""

parsed_data = parse_code_change_file(file_content)
print(parsed_data)

# To use with a file:
# with open('0158Table.java', 'r') as f:
#     file_content = f.read()
# parsed_data = parse_code_change_file(file_content)
# print(parsed_data)