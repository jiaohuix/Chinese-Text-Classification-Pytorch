from openmind_hub import upload_folder, create_repo

TOKEN = ""
# REPO_ID = "jiaohuix/cls26-02-swift"
REPO_ID = "camp_test_jhx25/cls26-02-swift"
MODEL_PATH = "/root/train-paper/code/swift_output/InternLM2.5-1.8B-Lora/merged"

create_repo(
    token=TOKEN,
    repo_id=REPO_ID,
    repo_type="model"
)

upload_folder(
    token=TOKEN,
    folder_path=MODEL_PATH,#模型位置
    repo_id=REPO_ID,#模型空间名字
)