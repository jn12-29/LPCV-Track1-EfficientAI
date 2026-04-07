import qai_hub as hub

# 1. 将这里替换为你实际的 Compile Job ID
# 你可以在 Qualcomm AI Hub 网页端的 "Jobs" 列表中找到这个 ID（通常以 'j' 开头，例如 'j1234abcd'）
job_id_list = ["jperv7v8g", "jpv1wlwzp"]  # text image

# 2. 获取该编译作业对象
for job_id in job_id_list:
    compile_job = hub.get_job(job_id)

    # 3. 添加共享权限
    compile_job.modify_sharing(add_emails=["lowpowervision@gmail.com"])

    print(f"作业 {job_id} 的权限已成功共享给 lowpowervision@gmail.com！")
