import os
import subprocess
from Bio import motifs

# 输入文件夹路径（每个文件是一个噬菌体基因组）
input_folder = "case_phage"
output_folder = "case_phage_motif"

os.makedirs(output_folder, exist_ok=True)

# 遍历每个基因组文件
for file in os.listdir(input_folder):
    if file.endswith(".fasta") or file.endswith(".fa"):
        phage_name = os.path.splitext(file)[0].replace(".unitigs",'') # 适用于case_study这个文件夹
        fasta_file = os.path.join(input_folder, file)

        meme_output = os.path.join(output_folder, f"{phage_name}_meme_out")
        motif_output = os.path.join(output_folder, f"{phage_name}_motifs.txt")

        # 如果基序信息文件已存在并且不为空，则跳过重新计算
        if os.path.exists(motif_output) and os.path.getsize(motif_output) > 0:
            print(f"跳过 {phage_name}，基序信息已存在：{motif_output}")
            continue

        # 创建 MEME 输出目录
        os.makedirs(meme_output, exist_ok=True)

        print(f"Processing {phage_name}...")

        # 调用 MEME 进行基序发现
        try:
            subprocess.run(
                [
                    "meme", fasta_file,
                    "-oc", meme_output,  # 输出目录
                    "-dna",  # 指定序列是 DNA
                    "-mod", "anr",  # "zoops" 模式：每个序列可能包含 0 或 1 个基序
                    "-nmotifs", "5",  # 最多发现 5 个基序
                    "-minw", "5",  # 基序的最小长度
                    "-maxw", "72"  # 基序的最大长度
                ],
                check=True
            )
            print(f"MEME 基序发现完成: {phage_name}")
        except Exception as e:
            print(f"MEME 基序发现失败: {phage_name}，错误信息: {e}")
            continue

        meme_motif_file = os.path.join(meme_output, "meme.xml")
        if not os.path.exists(meme_motif_file):
            print(f"未找到 {phage_name} 的 MEME 输出文件，请检查 MEME 是否正确运行。")
            continue

        try:
            with open(meme_motif_file) as handle:
                meme_motifs = motifs.parse(handle, "meme")
        except Exception as e:
            print(f"解析基序文件失败: {phage_name}，错误信息: {e}")
            continue

        with open(motif_output, "w") as outfile:
            for i, motif in enumerate(meme_motifs):
                outfile.write(f"Motif {i + 1}:\n")
                outfile.write(f"Consensus sequence: {motif.consensus}\n")
                outfile.write("Instances:\n")
                for instance in motif.instances:
                    outfile.write(f"  {instance.sequence_id} at {instance.start}\n")
                outfile.write("\n")
        print(f"基序信息保存到 {motif_output}")

print("所有基因组基序识别任务完成。")
