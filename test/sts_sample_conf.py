import logging
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials

sts_host = "http://sts.bj.baidubce.com"
##Bos 申请获取AK
access_key_id = "72f6f5c1b798429b956835c6a2d08515"

##Bos 申请获取SK
secret_access_key = "d99efd1abec6459997cd32e15254966b"

logger = logging.getLogger('baidubce.services.sts.stsclient')
fh = logging.FileHandler("sample.log")
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(fh)

config = BceClientConfiguration(credentials=BceCredentials(access_key_id, secret_access_key), endpoint=sts_host)

