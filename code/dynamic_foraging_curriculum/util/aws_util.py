import os
import logging

import boto3
from botocore.exceptions import ClientError

BUCKET = 'aind-behavior-data'

logger = logging.getLogger(__name__)

def export_df_and_upload(df,
                         file_name,
                         local_cache_path='/root/capsule/results/',
                         s3_path='foraging_auto_training/',
                         method='df_to_csv'):

    # save to local cache
    local_file_name = local_cache_path + file_name

    if method == 'df_to_pickle':
        df.to_pickle(local_file_name)
    elif method == 'df_to_csv':
        df.to_csv(local_file_name)
    elif method == 'dill_dump':
        dill.dump(df, open(local_file_name, 'wb'))

    size = os.path.getsize(local_file_name) / (1024 * 1024)

    # upload to s3
    s3_file_name = s3_path + file_name
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(local_file_name,
                              BUCKET,
                              s3_file_name)
        logger.info(f'file exported to {BUCKET}/{s3_file_name}, '
                    f'size = {size} MB, df_length = {len(df)}')
    except ClientError as e:
        logger.error(e)


if __name__ == '__main__':
    # test
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(np.random.randn(10, 5), columns=[
                      'a', 'b', 'c', 'd', 'e'])
    export_df_and_upload(df,
                         file_name='test.csv',
                         method='df_to_csv')
