import subprocess
import pandas as pd
import logging
import datetime
import sys
import os
import numpy as np
import argparse
import re
import json

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
directory = "/mnt/cephfs/home/ckoutlis/slurm"
logging.basicConfig(filename=os.path.join(directory, "monitoring.log"), filemode="w", level=logging.INFO, format=FORMAT)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

squeue_csv_path = os.path.join(directory, "monitoring.csv")
if save_squeue_csv != "y" and os.path.exists(squeue_csv_path):
    os.remove(squeue_csv_path)
logger.info(f"[{datetime.datetime.now()}] Slurm cluster monitoring - Save squeue .csv: {save_squeue_csv}")

basic_info_commands = """
echo /================        NODES INFO           ===============/
sinfo --format="%.10n | %.5t | %.7z | %.8e | %.7m | %.9O | %.13C | %G"
echo
echo /================        SQUEUE:ALL           ===============/
squeue --format="%.6A %.16i %.10j %.10u %.8T %.5c %.7m %.22b %.10M %.12l %.5D %.17R"
echo 
echo /================        SQUEUE:ME:RUNNING           ===============/
squeue --format="%.6A %.16i %.10j %.10u %.8T %.5c %.7m %.15b %.10M %.12l %.5D %.17R" --me -t RUNNING
echo 
echo /================        SQUEUE:ME:PENDING           ===============/
squeue --start --format="%.6A %.16i %.10j %.10u %.8T %.5c %.7m %.15b %.10M %.12l %.5D %.17R" --me -t PENDING
echo
"""
jobs = subprocess.check_output(basic_info_commands, shell=True)
for line in jobs.splitlines():
    logger.info(line.decode("UTF-8"))

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


# Per user
metrics = ["JOBS", "CPUS", "HOST_MEM_GB", "GPU_MEM_GB"]


def get_info_per_user(data):
    users = data["USER"].unique()
    df = pd.DataFrame(data=[["nan"] * len(metrics)] * len(users), columns=metrics, index=users)
    for user in users:
        df.loc[user, "JOBS"] = data[data["USER"] == user].shape[0]
        df.loc[user, "CPUS"] = sum([int(x) for x in data["CPUS"][data["USER"] == user].tolist()])
        df.loc[user, "HOST_MEM_GB"] = sum(
            [int(x.replace("G", "")) for x in data["MIN_MEMORY"][data["USER"] == user].tolist()]
        )
        sum_gpu_mem = 0
        for x in data["TRES_PER_NODE"][data["USER"] == user].tolist():
            if "gres/shard:" in x:
                sum_gpu_mem += int(
                    x[
                        x.rfind(
                            ":",
                        )
                        + 1 :
                    ]
                )
        df.loc[user, "GPU_MEM_GB"] = sum_gpu_mem
    return df


logger.info("\nAggregate used resources per user")
df_per_user = get_info_per_user(df_per_job)
logger.info(df_per_user.sort_values(by="GPU_MEM_GB", ascending=False))
logger.info("\nAggregate pending resources per user")
logger.info(get_info_per_user(df_per_job_pending).sort_values(by="GPU_MEM_GB", ascending=False))

# Per metric
node_up_states = ["mix", "alloc", "idle"]
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
        int(y[y.rfind(":") + 1 :])
        for x in df_sinfo[df_sinfo["STATE"].isin(node_up_states)]["GRES"].tolist()
        for y in x.split(",")
        if "shard" in y
    ]
)

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
nodes = df_per_job["NODELIST"].unique()
metrics = ["CPUS", "HOST_MEM_GB", "GPU_MEM_GB"]
df_per_node = pd.DataFrame(data=[["nan"] * len(metrics)] * len(nodes), columns=metrics, index=nodes)
sum_free_perc = []
free_gpu_mem_ = []
for node in nodes:
    # CPUS
    used_node_cpus = sum([int(x) for x in df_per_job["CPUS"][df_per_job["NODELIST"] == node].tolist()])
    tot_node_cpus = int(df_sinfo[df_sinfo["HOSTNAMES"] == node]["CPUS(A/I/O/T)"].values[0].split("/")[-1])
    free_node_cpus = tot_node_cpus - used_node_cpus
    free_perc_node_cpus = free_node_cpus / tot_node_cpus
    df_per_node.loc[node, "CPUS"] = (
        f"{used_node_cpus}/{free_node_cpus}/{tot_node_cpus} - {free_perc_node_cpus*100:1.1f}%"
    )

    # HOST MEM
    used_node_host_mem = sum(
        [int(x.replace("G", "")) for x in df_per_job["MIN_MEMORY"][df_per_job["NODELIST"] == node].tolist()]
    )
    tot_node_host_mem = int((df_sinfo[df_sinfo["HOSTNAMES"] == node]["MEMORY"].astype(int) / 1000).values[0])
    free_node_host_mem = tot_node_host_mem - used_node_host_mem
    free_perc_node_host_mem = free_node_host_mem / tot_node_host_mem
    df_per_node.loc[node, "HOST_MEM_GB"] = (
        f"{used_node_host_mem}/{free_node_host_mem}/{tot_node_host_mem} - {free_perc_node_host_mem*100:1.1f}%"
    )

    # GPU MEM
    used_node_gpu_mem = 0
    for x in df_per_job["TRES_PER_NODE"][df_per_job["NODELIST"] == node].tolist():
        if "gres/shard:" in x:
            used_node_gpu_mem += int(
                x[
                    x.rfind(
                        ":",
                    )
                    + 1 :
                ]
            )
    tot_node_gpu_mem = sum(
        [
            int(y[y.rfind(":") + 1 :])
            for y in df_sinfo[df_sinfo["HOSTNAMES"] == node]["GRES"].values[0].split(",")
            if "shard" in y
        ]
    )
    free_node_gpu_mem = tot_node_gpu_mem - used_node_gpu_mem
    free_perc_node_gpu_mem = free_node_gpu_mem / tot_node_gpu_mem
    df_per_node.loc[node, "GPU_MEM_GB"] = (
        f"{used_node_gpu_mem}/{free_node_gpu_mem}/{tot_node_gpu_mem} - {free_perc_node_gpu_mem*100:1.1f}%"
    )

    sum_free_perc.append(free_perc_node_cpus + free_perc_node_host_mem + free_perc_node_gpu_mem)
    free_gpu_mem_.append(free_node_gpu_mem)

sort_nodes_by_ = "gpu"
assert sort_nodes_by_ in ["perc", "gpu"]
arg = np.argsort(sum_free_perc if sort_nodes_by_ == "perc" else free_gpu_mem_)
df_per_node = df_per_node.reindex(index=[nodes[i] for i in arg])
logger.info(f"\nAggregate used resources per node (USED/FREE/TOTAL - FREE PERC.) SORTED BY {sort_nodes_by_.upper()}")
logger.info(df_per_node)


def get_job_info(jobid):
    job = subprocess.check_output(f"scontrol show job -d {jobid}", shell=True)
    info = dict(
        [z.split("=", 1) for y in job.decode("utf-8").split("\n") for z in y.split(" ") if "=" in z]
    )  # TODO: change with regular expression, when job name has space (" ") it crashes
    return info


# Per GPU device
logger.info(f"\nAllocated memory per GPU device (USED/FREE/TOTAL)")
running_jobids = df_per_job["JOBID"].to_numpy()[:, 0]
shards = {}
for jobid in running_jobids:
    info = get_job_info(jobid)
    node = info["Nodes"]
    devices = [device if device[-1] != "," else device[:-1] for device in info["GRES"].split("shard:")[1:]]
    if node not in shards:
        shards[node] = {}
    shards_d0 = 0
    shards_d1 = 0
    for i, text in enumerate(devices):
        match = re.match(r"(\w+):(\d+)\((\d+)/(\d+),(\d+)/(\d+)\)", text)
        if match:
            device_name = match.group(1)
            shards_d0 += int(match.group(3))
            shards_d1 += int(match.group(5))
            if f"device_{i}" not in shards[node]:
                shards[node][f"device_{i}"] = {"name": device_name, "total": int(match.group(4 + i * 2))}
            if "," in text and len(devices) == 1 and "device_1" not in shards[node]:
                shards[node]["device_1"] = {"name": device_name, "total": int(match.group(6))}
        else:
            match = re.match(r"(\w+):(\d+)\((\d+)/(\d+)\)", text)
            if match:
                device_name = match.group(1)
                shards_d0 += int(match.group(3))
                if f"device_{i}" not in shards[node]:
                    shards[node][f"device_{i}"] = {"name": device_name, "total": int(match.group(4))}
    if "device_0" in shards[node] and "used" not in shards[node]["device_0"]:
        shards[node]["device_0"]["used"] = shards_d0
        if "device_1" in shards[node] and "used" not in shards[node]["device_1"]:
            shards[node]["device_1"]["used"] = shards_d1
    elif "device_0" in shards[node] and "used" in shards[node]["device_0"]:
        shards[node]["device_0"]["used"] += shards_d0
        if "device_1" in shards[node] and "used" in shards[node]["device_1"]:
            shards[node]["device_1"]["used"] += shards_d1
data = []
for node in shards:
    for i, device in enumerate(shards[node]):
        shards[node][device]["free"] = shards[node][device]["total"] - shards[node][device]["used"]
        data.append(
            [
                node,
                f"{shards[node][device]['name']} ({i})",
                shards[node][device]["used"],
                shards[node][device]["free"],
                shards[node][device]["total"],
            ]
        )
df_per_gpu = pd.DataFrame(data=data, columns=["node", "device", "used", "free", "total"])
logger.info(df_per_gpu)
