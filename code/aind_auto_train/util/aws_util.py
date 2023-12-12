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


def download_dir_from_s3(bucket,
                         s3_path,
                         local_cache_path=None
                         ):
    """
    Download the contents of a folder directory
    Args:
        bucket: the name of the s3 bucket
        s3_path: the folder path in the s3 bucket
        local_cache_path: a relative or absolute directory path in the local file system
    https://stackoverflow.com/questions/49772151/download-a-folder-from-s3-using-boto3
    """
    bucket = s3_resource.Bucket(bucket)
    for obj in bucket.objects.filter(Prefix=s3_path):
        target = obj.key if local_cache_path is None \
            else os.path.join(local_cache_path, os.path.relpath(obj.key, s3_path))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target), exist_ok=True)
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)


if __name__ == '__main__':
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
