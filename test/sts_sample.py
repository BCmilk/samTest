import sts_sample_conf
from baidubce.services.sts.sts_client import StsClient
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials

sts_client = StsClient(sts_sample_conf.config)

duration_seconds = 3600
# you can specify limited permissions with ACL
access_dict = {}
access_dict["service"] = "bce:bos"
access_dict["region"] = "bj"   
access_dict["effect"] = "Allow"
resource = ["*"]
access_dict["resource"] = resource
permission = ["WRITE","READ"]
access_dict["permission"] = permission

access_control_list = {"accessControlList": [access_dict]}
# 新建StsClient
response = sts_client.get_session_token(acl=access_control_list, duration_seconds=duration_seconds)

sts_ak = str(response.access_key_id)
sts_sk = str(response.secret_access_key)
token = response.session_token
bos_host = "https://testyiyan.cd.bcebos.com"
#配置BceClientConfiguration
config = BceClientConfiguration(credentials=BceCredentials(sts_ak, sts_sk), endpoint = bos_host, security_token=token)

