import subprocess
import pandas as pd
import logging
import datetime
import sys
import os
import numpy as np
import argparse
import re
from pathlib import Path


def expand_nodelist(nodelist_str):
    if "[" not in nodelist_str:
        return [nodelist_str]
    base_name, bracket_part = nodelist_str.split("[")
    ranges_str = bracket_part.rstrip("]")
    parts = ranges_str.split(",")
    nodes = []
    for part in parts:
        if "-" in part:
            try:
                start, end = map(int, part.split("-"))
                for i in range(start, end + 1):
                    nodes.append(f"{base_name}{i}")
            except ValueError:
                nodes.append(nodelist_str)
        else:
            nodes.append(f"{base_name}{part}")
    return nodes


parser = argparse.ArgumentParser(description="Experiments")
parser.add_argument(
    "-s",
    "--save",
    help="y or n: save squeue to .csv",
    choices=["y", "n"],
)
args = parser.parse_args()
save_squeue_csv = args.save

logger = logging.getLogger()
FORMAT = "%(message)s"
directory = os.path.join(Path.home(), "slurm")
logging.basicConfig(filename=os.path.join(directory, "monitoring.log"), filemode="w", level=logging.INFO, format=FORMAT)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

gpu_type_mem = {"rtx_4090": 24, "rtx_3060": 12, "rtx_3090_ti": 24, "rtx_4080": 16, "gtx_1080": 8}

squeue_csv_path = os.path.join(directory, "monitoring.csv")
if save_squeue_csv != "y" and os.path.exists(squeue_csv_path):
    os.remove(squeue_csv_path)
logger.info(f"[{datetime.datetime.now()}] Slurm cluster monitoring - Save squeue .csv: {save_squeue_csv}")

basic_info_commands = """
echo /================        NODES INFO           ===============/
sinfo --format="%.10n | %.5t | %.7z | %.8e | %.7m | %.9O | %.13C | %G"
echo
echo /================        SQUEUE:ALL           ===============/
squeue --format="%.6A %.16i %.10j %.10u %.3t %.5c %.7m %.26b %.16S %.10M %.12l %.5D %.17R"
echo 
echo /================        SQUEUE:ME:RUNNING           ===============/
squeue --format="%.6A %.16i %.10j %.10u %.3t %.5c %.7m %.26b %.16S %.10M %.12l %.5D %.17R" --me -t RUNNING
echo 
echo /================        SQUEUE:ME:PENDING           ===============/
squeue --start --format="%.6A %.16i %.10j %.10u %.3t %.5c %.7m %.26b %.16S %.11Y %.12l %.5D %.17R" --me -t PENDING
echo
echo /================        SPRIO           ===============/
sprio -o "%.15i %.8u %.15o %.10Y %.10F %.10n %.10Q"
echo
echo /================        SSHARE           ===============/
sshare -a | head -n 2
sshare -a | tail -n +6 | sort -k 7 -rn
echo
echo /================        MAX COMPUTE:ME           ===============/
sacctmgr show assoc tree format=account,user,grptres%50 user=$(whoami)
echo
echo /================        USED/MAX STORAGE:ME           ===============/
"""
jobs = subprocess.check_output(basic_info_commands, shell=True)
for line in jobs.splitlines():
    logger.info(line.decode("UTF-8"))

max_storage_results = subprocess.check_output(
    "getfattr -n ceph.quota.max_bytes /mnt/cephfs/home/$(whoami)", shell=True, stderr=subprocess.DEVNULL
)
for line in max_storage_results.splitlines():
    match = re.match(r'ceph.quota.max_bytes="(\d+)"', line.decode("UTF-8"))
    if match:
        max_storage, max_measure = int(match.group(1)) / 1024**3, "GB"
        max_storage, max_measure = (max_storage / 1024, "TB") if max_storage >= 1000 else (max_storage, "GB")
used_storage_results = subprocess.check_output(
    "getfattr -n ceph.dir.rbytes /mnt/cephfs/home/$(whoami)", shell=True, stderr=subprocess.DEVNULL
)
for line in used_storage_results.splitlines():
    match = re.match(r'ceph.dir.rbytes="(\d+)"', line.decode("UTF-8"))
    if match:
        used_storage, used_measure = int(match.group(1)) / 1024 ** (3 if max_measure == "GB" else 4), (
            "GB" if max_measure == "GB" else "TB"
        )
logger.info(f"{used_storage:1.1f} {used_measure} / {max_storage:1.1f} {max_measure}")
if used_storage >= max_storage:
    logger.info("Storage limit reached/exceeded!")
else:
    logger.info(f"Free storage: {(max_storage - used_storage):1.1f} {max_measure}")

# Per job: running
squeue_command = "squeue --format='%all'"
jobs = subprocess.check_output(f"{squeue_command} -t RUNNING", shell=True)
lines = jobs.splitlines()
data = []
columns = lines[0].decode("UTF-8").split("|")
for line in lines[1:]:
    data.append(line.decode("UTF-8").split("|"))
df_per_job = pd.DataFrame(data=data, columns=columns)
if save_squeue_csv == "y":
    df_per_job.to_csv(squeue_csv_path)

# Per job: pending
jobs = subprocess.check_output(f"{squeue_command} --array -t PENDING", shell=True)
lines = jobs.splitlines()
data_pending = []
columns = lines[0].decode("UTF-8").split("|")
for line in lines[1:]:
    data_pending.append(line.decode("UTF-8").split("|"))
df_per_job_pending = pd.DataFrame(data=data_pending, columns=columns)

# Per metric
node_up_states = ["mix-", "mix", "alloc", "idle", "drng"]
sinfo_command = "sinfo --format='%.10n | %.5t | %.7z | %.8e | %.7m | %.9O | %.13C | %G'"
sinfo = subprocess.check_output(sinfo_command, shell=True)
lines = sinfo.splitlines()
data = []
columns = [x.replace(" ", "") for x in lines[0].decode("UTF-8").split("|")]
for line in lines[1:]:
    data.append([x.replace(" ", "") for x in line.decode("UTF-8").split("|")])
df_sinfo = pd.DataFrame(data=data, columns=columns)
tot_cpus = sum(
    [int(x.split("/")[-1]) for x in df_sinfo[df_sinfo["STATE"].isin(node_up_states)]["CPUS(A/I/O/T)"].tolist()]
)
tot_host_mem = int(df_sinfo[df_sinfo["STATE"].isin(node_up_states)]["MEMORY"].astype(int).sum() / 1000)
tot_gpu_mem = sum(
    [
        int(y[y.find("=" if "=" in y else ":", 6) + 1 : y.rfind("(")])
        for x in df_sinfo[df_sinfo["STATE"].isin(node_up_states)]["GRES"].tolist()
        for y in x.split(",")
        if "shard" in y
    ]
)
# Break: it will continue after Per user...

# Per user
metrics = ["JOBS", "CPUS", "HOST_MEM_GB", "GPU_MEM_GB"]


def get_info_per_user(data):
    users = data["USER"].unique()
    df = pd.DataFrame(data=[["nan"] * len(metrics)] * len(users), columns=metrics, index=users)
    for user in users:
        df.loc[user, "JOBS"] = data[data["USER"] == user].shape[0]
        df.loc[user, "CPUS"] = sum([int(x) for x in data["CPUS"][data["USER"] == user].tolist()])
        df.loc[user, "HOST_MEM_GB"] = sum(
            [int(x[0].replace("G", "")) * int(x[1]) for x in data[["MIN_MEMORY", "NODES"]][data["USER"] == user].values]
        )
        sum_gpu_mem = 0
        for x, x_nodes in data[["TRES_PER_NODE", "NODES"]][data["USER"] == user].values:
            if "gres/shard" in x:
                sum_gpu_mem += int(
                    x[
                        x.rfind(
                            "=" if "=" in x else ":",
                        )
                        + 1 :
                    ]
                )
            elif "gres/gpu" in x:
                gpu_type_num = x[x.find("gres/gpu") + 9 :]
                if ":" in gpu_type_num:
                    gpu_type, gpu_num = gpu_type_num.split(":")
                else:
                    gpu_type = gpu_type_num
                    gpu_num = 1
                sum_gpu_mem += gpu_type_mem[gpu_type] * int(gpu_num) * int(x_nodes)
        df.loc[user, "GPU_MEM_GB"] = sum_gpu_mem
    tot_jobs = df["JOBS"].sum()
    jobs_p = []
    cpus_p = []
    host_mem_gb_p = []
    gpu_mem_gb_p = []
    for user in users:
        jobs_p.append(f"{df.loc[user, 'JOBS']/tot_jobs*100:1.1f}%")
        cpus_p.append(f"{df.loc[user, 'CPUS']/tot_cpus*100:1.1f}%")
        host_mem_gb_p.append(f"{df.loc[user, 'HOST_MEM_GB']/tot_host_mem*100:1.1f}%")
        gpu_mem_gb_p.append(f"{df.loc[user, 'GPU_MEM_GB']/tot_gpu_mem*100:1.1f}%")
    df.insert(1, "JOBS_P", jobs_p)
    df.insert(3, "CPUS_P", cpus_p)
    df.insert(5, "HOST_MEM_GB_P", host_mem_gb_p)
    df.insert(7, "GPU_MEM_GB_P", gpu_mem_gb_p)
    return df


logger.info("\nAggregate used resources per user")
df_per_user = get_info_per_user(df_per_job)
logger.info(df_per_user.sort_values(by="GPU_MEM_GB", ascending=False))
logger.info("\nAggregate pending resources per user")
logger.info(get_info_per_user(df_per_job_pending).sort_values(by="GPU_MEM_GB", ascending=False))

# Continue from Per metric...
df_per_metric = pd.DataFrame(columns=metrics[1:])
used = df_per_user[metrics[1:]].sum().values
used_cpus = used[0]
used_host_mem = used[1]
used_gpu_mem = used[2]
free_cpus = tot_cpus - used_cpus
free_host_mem = tot_host_mem - used_host_mem
free_gpu_mem = tot_gpu_mem - used_gpu_mem
free_perc_cpus = free_cpus / tot_cpus
free_perc_host_mem = free_host_mem / tot_host_mem
free_perc_gpu_mem = free_gpu_mem / tot_gpu_mem
state = [
    f"{used_cpus}/{free_cpus}/{tot_cpus} - {free_perc_cpus*100:1.1f}%",
    f"{used_host_mem}/{free_host_mem}/{tot_host_mem} - {free_perc_host_mem*100:1.1f}%",
    f"{used_gpu_mem}/{free_gpu_mem}/{tot_gpu_mem} - {free_perc_gpu_mem*100:1.1f}%",
]
df_per_metric.loc["state"] = state
logger.info("\nAggregate used resources (USED/FREE/TOTAL - FREE PERC.)")
logger.info(df_per_metric)

# Per node
metrics = ["CPUS", "HOST_MEM_GB", "GPU_MEM_GB"]
all_running_nodes = sorted(
    list(set(node for nodes_str in df_per_job["NODELIST"] for node in expand_nodelist(nodes_str)))
)
df_per_node = pd.DataFrame(
    data=[["nan"] * len(metrics)] * len(all_running_nodes), columns=metrics, index=all_running_nodes
)

sum_free_perc = []
free_gpu_mem_ = []

for node in all_running_nodes:
    # Find all jobs running on this specific node
    jobs_on_node = df_per_job[df_per_job["NODELIST"].apply(lambda x: node in expand_nodelist(x))]

    # Get total resources for this node
    tot_node_info = df_sinfo[df_sinfo["HOSTNAMES"] == node]
    if tot_node_info.empty:
        # Handle case where a node in a running job is not found in sinfo
        df_per_node.loc[node] = "N/A"
        continue

    # Calculate total allocated resources on this node from all jobs
    total_used_node_cpus = 0
    total_used_node_host_mem = 0
    total_used_node_gpu_mem = 0

    for _, job_row in jobs_on_node.iterrows():
        expanded_nodes = expand_nodelist(job_row["NODELIST"])
        num_nodes_in_job = len(expanded_nodes)

        # Assuming resources are distributed evenly across nodes in a job
        total_used_node_cpus += int(job_row["CPUS"]) / num_nodes_in_job
        total_used_node_host_mem += int(job_row["MIN_MEMORY"].replace("G", ""))

        sum_gpu_mem_job = 0
        gres_str = job_row["TRES_PER_NODE"]
        if "gres/shard" in gres_str:
            sum_gpu_mem_job += int(gres_str[gres_str.rfind("=" if "=" in gres_str else ":") + 1 :])
        elif "gres/gpu" in gres_str:
            gpu_type_num = gres_str[gres_str.find("gres/gpu") + 9 :]
            if ":" in gpu_type_num:
                gpu_type, gpu_num = gpu_type_num.split(":")
            else:
                gpu_type = gpu_type_num
                gpu_num = 1
            sum_gpu_mem_job += gpu_type_mem[gpu_type] * int(gpu_num)

        total_used_node_gpu_mem += sum_gpu_mem_job

    # Get total resources for the node
    tot_node_cpus = int(tot_node_info["CPUS(A/I/O/T)"].values[0].split("/")[-1])
    tot_node_host_mem = int((tot_node_info["MEMORY"].astype(int) / 1000).values[0])
    tot_node_gpu_mem = (
        sum(
            [
                int(y[y.find("=" if "=" in y else ":", 6) + 1 : y.rfind("(")])
                for y in tot_node_info["GRES"].values[0].split(",")
                if "shard" in y
            ]
        )
        if "shard" in tot_node_info["GRES"].values[0]
        else 0
    )

    # Calculate free resources and percentages
    free_node_cpus = tot_node_cpus - total_used_node_cpus
    free_node_host_mem = tot_node_host_mem - total_used_node_host_mem
    free_node_gpu_mem = tot_node_gpu_mem - total_used_node_gpu_mem

    free_perc_cpus = free_node_cpus / tot_node_cpus if tot_node_cpus > 0 else 0
    free_perc_host_mem = free_node_host_mem / tot_node_host_mem if tot_node_host_mem > 0 else 0
    free_perc_gpu_mem = free_node_gpu_mem / tot_node_gpu_mem if tot_node_gpu_mem > 0 else 0

    df_per_node.loc[node, "CPUS"] = (
        f"{int(total_used_node_cpus)}/{int(free_node_cpus)}/{tot_node_cpus} - {free_perc_cpus*100:1.1f}%"
    )
    df_per_node.loc[node, "HOST_MEM_GB"] = (
        f"{int(total_used_node_host_mem)}/{int(free_node_host_mem)}/{tot_node_host_mem} - {free_perc_host_mem*100:1.1f}%"
    )
    df_per_node.loc[node, "GPU_MEM_GB"] = (
        f"{int(total_used_node_gpu_mem)}/{int(free_node_gpu_mem)}/{tot_node_gpu_mem} - {free_perc_gpu_mem*100:1.1f}%"
    )

    sum_free_perc.append(free_perc_cpus + free_perc_host_mem + free_perc_gpu_mem)
    free_gpu_mem_.append(free_node_gpu_mem)

sort_nodes_by_ = "gpu"
assert sort_nodes_by_ in ["perc", "gpu"]
arg = np.argsort(sum_free_perc if sort_nodes_by_ == "perc" else free_gpu_mem_)
df_per_node = df_per_node.reindex(index=[all_running_nodes[i] for i in arg])
logger.info(f"\nAggregate used resources per node (USED/FREE/TOTAL - FREE PERC.) SORTED BY {sort_nodes_by_.upper()}")
logger.info(df_per_node)


def get_job_info(jobid):
    job = subprocess.check_output(f"scontrol show job -d {jobid}", shell=True)
    info = dict([z.split("=", 1) for y in job.decode("utf-8").split("\n") for z in y.split(" ") if "=" in z])
    return info


# Per GPU device


def create_gpu_dataframe(data_dict):
    rows = []
    for node, devices in data_dict.items():
        if devices:
            for device_index, device_info in devices.items():
                device_name = device_info.get("name")
                used_mem = device_info.get("used")
                total_mem = device_info.get("total")
                free_mem = total_mem - used_mem
                row = {
                    "node": node,
                    "device-index": device_index,
                    "device-name": device_name,
                    "used": used_mem,
                    "free": free_mem,
                    "total": total_mem,
                }
                rows.append(row)
    df = pd.DataFrame(rows)
    return df


logger.info(f"\nAllocated memory per GPU device (USED/FREE/TOTAL)")
running_jobids = df_per_job["JOBID"].to_numpy()[:, 0]
shards = {}
for jobid in running_jobids:
    info = get_job_info(jobid)
    nodes = expand_nodelist(info["Nodes"])
    devices = [
        device if device[-1] != "," else device[:-1]
        for device in info["GRES"].split("shard:" if "shard:" in info["GRES"] else "gpu:")[1:]
    ]
    for node in nodes:
        if node not in shards:
            shards[node] = {}
        for i, text in enumerate(devices):
            match = re.match(r"(\w+):(\d+)", text[: text.find("(")])
            device_name = match.group(1)
            if "IDX" in text[text.find("(") :]:
                shards_current = int(match.group(2))
                match = re.match(r"\(IDX:(0-1|0|1)\)", text[text.find("(") :])
                indices = [int(x) for x in match.group(1) if x != "-"]
                for device_index in indices:
                    shards[node][device_index] = {
                        "used": shards_current * gpu_type_mem[device_name] // len(indices),
                        "total": gpu_type_mem[device_name],
                        "name": device_name,
                    }
            else:
                shards_current = int(match.group(2))
                if shards_current > 0:
                    match = re.match(r"\((\d+)/(\d+),(\d+)/(\d+)\)", text[text.find("(") :])
                    device_index = np.where(np.array([int(match.group(1)), int(match.group(3))]) == shards_current)[0][
                        0
                    ]
                    if device_index in shards[node]:
                        shards[node][device_index]["used"] += shards_current
                    else:
                        shards[node][device_index] = {
                            "used": shards_current,
                            "total": gpu_type_mem[device_name],
                            "name": device_name,
                        }

print(create_gpu_dataframe(shards))
