import os
import logging

import boto3
from botocore.exceptions import ClientError

BUCKET = 'aind-behavior-data'

logger = logging.getLogger(__name__)

s3_client = boto3.client('s3')


def export_df_and_upload(df,
                         file_name,
                         local_cache_path='/root/capsule/results/',
                         s3_path='foraging_auto_training/',
                         method='csv'):

    # save to local cache
    local_file_name = local_cache_path + file_name

    if method == 'pickle':
        df.to_pickle(local_file_name)
    elif method == 'csv':
        df.to_csv(local_file_name)

    size = os.path.getsize(local_file_name) / (1024 * 1024)

    # upload to s3
    s3_file_name = s3_path + file_name
    try:
        s3_client.upload_file(local_file_name,
                              BUCKET,
                              s3_file_name)
        logger.info(f'file exported to {BUCKET}/{s3_file_name}, '
                    f'size = {size} MB, df_length = {len(df)}')
    except ClientError as e:
        logger.error(e)


def download_and_import_df(file_name,
                           local_cache_path='/root/capsule/results/',
                           s3_path='foraging_auto_training/',
                           method='csv'):

    # download from s3
    s3_file_name = s3_path + file_name
    local_file_name = local_cache_path + file_name

    try:
        s3_client.download_file(BUCKET,
                                s3_file_name,
                                local_file_name)

        size = os.path.getsize(local_file_name) / (1024 * 1024)
        logger.info(f'file downloaded to {BUCKET}/{s3_file_name}, '
                    f'size = {size} MB, df_length = {len(df)}')
    except FileNotFoundError as e:
        logger.info(f'file not found: {e}')
        return e
    except ClientError as e:
        logger.error(e)

    if method == 'pickle':
        return pd.read_pickle(local_file_name)
    elif method == 'csv':
        return pd.read_csv(local_file_name)


if __name__ == '__main__':
    # test
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(np.random.randn(10, 5), columns=[
                      'a', 'b', 'c', 'd', 'e'])
    export_df_and_upload(df,
                         file_name='test.csv',
                         local_cache_path='/root/capsule/results/',
                         s3_path='foraging_auto_training/',
                         method='csv')

    df = download_and_load_df(file_name='test.csv',
                              local_cache_path='/root/capsule/results/',
                              s3_path='foraging_auto_training/',
                              method='csv')

    print(df)
