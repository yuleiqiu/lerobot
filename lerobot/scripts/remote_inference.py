import paramiko
import pickle
import numpy as np

def remote_inference(observation, remote_host, username, password=None, key_filename=None):
    # 将 observation 序列化
    serialized_obs = pickle.dumps(observation)
    
    # 创建 SSH 客户端，并建立连接
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(remote_host, username=username, password=password, key_filename=key_filename)
    
    # 使用 SFTP 上传 observation 文件到远程服务器（例如存放在 /tmp 目录下）
    sftp = ssh.open_sftp()
    remote_obs_file = '/tmp/observation.pkl'
    remote_action_file = '/tmp/action.pkl'
    with sftp.open(remote_obs_file, 'wb') as f:
        f.write(serialized_obs)
    sftp.close()
    
    # 远程执行推理脚本（确保服务器上该脚本的路径正确）
    # 例如：远程脚本在 /home/username/remote_inference_script.py
    command = f"python3 /home/{username}/remote_inference_script.py {remote_obs_file} {remote_action_file}"
    stdin, stdout, stderr = ssh.exec_command(command)
    err = stderr.read()
    if err:
        raise Exception("远程推理出错: " + err.decode())
    
    # 下载生成的 action 文件
    sftp = ssh.open_sftp()
    with sftp.open(remote_action_file, 'rb') as f:
        serialized_action = f.read()
    sftp.close()
    ssh.close()
    
    # 反序列化得到 action 数组
    action = pickle.loads(serialized_action)
    return action

# test
if __name__ == '__main__':
    # dummy observations
    observation = {
        "observation.images.cam_high": np.random.rand(3, 480, 640).astype(np.float32),
        "observation.images.cam_left_wrist": np.random.rand(3, 480, 640).astype(np.float32),
        "observation.images.cam_right_wrist": np.random.rand(3, 480, 640).astype(np.float32),
        "observation.state": np.random.rand(16).astype(np.float32)
    }
    remote_host = "remote.server.address"
    username = "your_username"
    # 如果使用密钥认证，则可设置 key_filename 参数
    action = remote_inference(observation, remote_host, username, password="your_password")
    print("从服务器返回的 action:", action)
