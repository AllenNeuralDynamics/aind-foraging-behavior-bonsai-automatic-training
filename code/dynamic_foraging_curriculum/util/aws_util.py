import os
import logging
import configparser

import pandas as pd
import boto3
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)


def get_aws_credentials(profile='default'):
    """Explicitly get AWS credentials for boto3 client 
    (somehow boto3.client('s3') doesn't work out of box)
    First check if the credentials are in the environment variables.
    If not, try ~/.aws/credentials (for windows, %UserProfile%\.aws\credentials)
        The content of the file should look like this:

        [default]
        AWS_ACCESS_KEY_ID=foo
        AWS_SECRET_ACCESS_KEY=bar

        See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    :param profile: The profile to read from the credentials file. Default is 'default'.
    :return: A dictionary containing 'aws_access_key_id' and 'aws_secret_access_key', or None if not found.
    """

    # --- Try environment variables first ---
    if 'AWS_SECRET_ACCESS_KEY' in os.environ and 'AWS_ACCESS_KEY_ID' in os.environ:
        logger.info(f'Found AWS credential from environment variables!')
        return {
            'aws_access_key_id': os.environ['AWS_ACCESS_KEY_ID'],
            'aws_secret_access_key': os.environ['AWS_SECRET_ACCESS_KEY']
        }

    # --- Try reading from ~/.aws/credentials ---
    # Construct the path to the credentials file
    credentials_path = os.path.expanduser("~/.aws/credentials")

    # Check if credentials file exists
    if not os.path.exists(credentials_path):
        logger.error("AWS credentials file not found.")
        return None

    # Read the credentials file
    config = configparser.ConfigParser()
    config.read(credentials_path)

    # Retrieve credentials for the specified profile
    if profile in config:
        aws_access_key_id = config[profile].get('aws_access_key_id')
        aws_secret_access_key = config[profile].get('aws_secret_access_key')
        logger.info(f'Found AWS credential from ~/.aws/credentials!')
        return {
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key
        }

    logger.error(f"Profile '{profile}' not found in credentials file.")
    return None


aws_credentials = get_aws_credentials()
s3_client = boto3.client('s3',
                         aws_access_key_id=aws_credentials['aws_access_key_id'],
                         aws_secret_access_key=aws_credentials['aws_secret_access_key']
                         )


def export_and_upload_df(df,
                         file_name,
                         local_cache_path='/root/capsule/results/',
                         bucket='aind-behavior-data',
                         s3_path='foraging_auto_training/'):

    # save to local cache
    local_file_name = local_cache_path + file_name

    os.makedirs(local_cache_path, exist_ok=True)

    if file_name.split('.')[-1] == 'pkl':
        df.to_pickle(local_file_name)
    elif file_name.split('.')[-1] == 'csv':
        df.to_csv(local_file_name)

    size = os.path.getsize(local_file_name) / (1024 * 1024)

    # upload to s3
    s3_file_name = s3_path + file_name
    try:
        s3_client.upload_file(local_file_name,
                              bucket,
                              s3_file_name)
        logger.info(f'file exported to {bucket}/{s3_file_name}, '
                    f'size = {size} MB, df_length = {len(df)}')
    except ClientError as e:
        logger.error(e)


def download_and_import_df(file_name,
                           bucket='aind-behavior-data',
                           s3_path='foraging_auto_training/',
                           local_cache_path='/root/capsule/results/',
                           ):

    # download from s3
    s3_file_name = s3_path + file_name
    local_file_name = local_cache_path + file_name

    os.makedirs(local_cache_path, exist_ok=True)

    try:
        s3_client.download_file(bucket,
                                s3_file_name,
                                local_file_name)

        size = os.path.getsize(local_file_name) / (1024 * 1024)
    except ClientError as e:
        logger.warning(f's3://{bucket}/{s3_path}{file_name} not found!')
        return None

    # import to df
    if file_name.split('.')[-1] == 'pkl':
        df = pd.read_pickle(local_file_name)
    elif file_name.split('.')[-1] == 'csv':
        df = pd.read_csv(local_file_name)

    logger.info(f'file downloaded to {bucket}/{s3_file_name}, '
                f'size = {size} MB, df_length = {len(df)}')

    return df


if __name__ == '__main__':
    # test
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(np.random.randn(10, 5), columns=[
                      'a', 'b', 'c', 'd', 'e'])
    export_and_upload_df(df,
                         file_name='test.csv',
                         local_cache_path='/root/capsule/results/',
                         s3_path='foraging_auto_training/',
                         )

    df = download_and_import_df(file_name='test.csv',
                                local_cache_path='/root/capsule/results/',
                                s3_path='foraging_auto_training/',
                                )

    print(df)
