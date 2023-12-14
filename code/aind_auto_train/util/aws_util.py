import os
import logging
import configparser

import pandas as pd
import s3fs


logger = logging.getLogger(__name__)


def get_aws_credentials(profile='default'):
    """Explicitly get AWS credentials
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

    logger.info(
        f'AWS credentials not found in environment variables. Try ~/.aws/credentials...')

    # --- Try reading from ~/.aws/credentials ---
    # Construct the path to the credentials file
    credentials_path = os.path.expanduser("~/.aws/credentials")

    # Check if credentials file exists
    if not os.path.exists(credentials_path):
        logger.error(
            "AWS credentials file not found at ~/.aws/credentials either!")
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


# Setup s3fs filesystem
aws_credentials = get_aws_credentials()
fs = s3fs.S3FileSystem(key=aws_credentials['aws_access_key_id'],
                       secret=aws_credentials['aws_secret_access_key'])


# Function to export DataFrame to S3
def export_df_to_s3(df,
                    file_name,
                    bucket='aind-behavior-data',
                    s3_path='foraging_auto_training/'
                    ):
    try:
        s3_file_path = f"{bucket}/{s3_path}{file_name}"
        if file_name.endswith('.pkl'):
            df.to_pickle(fs.open(s3_file_path, 'wb'))
        elif file_name.endswith('.csv'):
            df.to_csv(fs.open(s3_file_path, 'w'))

        logger.info(f'Dataframe exported to s3://{s3_file_path}, '
                    f'len(df) = {len(df)}')
    except OSError as e:
        logger.error(f'Error writing file to S3: {e}')


def import_df_from_s3(file_name,
                      bucket='aind-behavior-data',
                      s3_path='foraging_auto_training/'
                      ):

    s3_file_path = f"{bucket}/{s3_path}{file_name}"
    try:
        if file_name.endswith('.pkl'):
            df = pd.read_pickle(fs.open(s3_file_path, 'rb'))
        elif file_name.endswith('.csv'):
            df = pd.read_csv(fs.open(s3_file_path))
        logger.info(f'Dataframe imported from s3://{s3_file_path}, '
                    f'len(df) = {len(df)}')
        return df
    except FileNotFoundError:
        logger.error(f'File not found: s3://{s3_file_path}')
        return None
    except OSError as e:
        logger.error(f'Error reading file from S3: {e}')
        return None


def download_dir_from_s3(bucket='aind-behavior-data',
                         s3_dir='foraging_auto_training/saved_curriculums/',
                         local_dir='/root/capsule/scratch/saved_curriculums/',
                         ):
    """
    Copy a directory from S3 to local
    """
    s3_dir_path = f"{bucket}/{s3_dir}"
    try:
        res = fs.get(s3_dir_path, local_dir, recursive=True)
        logger.info(f'{len(res)-2} objects downloaded from s3://{s3_dir_path} '
                    f'to {local_dir}')
    except FileNotFoundError:
        logger.error(f'Directory not found: s3://{s3_file_path}')
        return None


def upload_dir_to_s3(local_dir='/root/capsule/scratch/saved_curriculums/',
                     bucket='aind-behavior-data',
                     s3_dir='foraging_auto_training/saved_curriculums/',
                     ):
    """
    Copy a directory from local to S3
    """
    s3_dir_path = f"{bucket}/{s3_dir}"
    try:
        res = fs.put(local_dir, s3_dir_path, recursive=True)
        logger.info(f'{len(res)} objects uploaded from {local_dir} '
                    f'to s3://{s3_dir_path} ')
    except FileNotFoundError:
        logger.error(f'Directory not found: {local_dir}')
        return None


if __name__ == '__main__':
    logger.addHandler(logging.StreamHandler())
    logger.setLevel('DEBUG')

    # test
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(np.random.randn(10, 5), columns=[
                      'a', 'b', 'c', 'd', 'e'])
    export_df_to_s3(df,
                    file_name='test.csv',
                    s3_path='foraging_auto_training/',
                    )

    df = import_df_from_s3(file_name='test.csv',
                           s3_path='foraging_auto_training/',
                           )

    print(df)

    download_dir_from_s3()
    upload_dir_to_s3()