import s3fs
import os

# Initialize the S3 filesystem
fs = s3fs.S3FileSystem(anon=False)  # anon=False for authenticated access

src_prefix = 'aind-behavior-data/foraging_auto_training/'
dst_prefix = 'aind-behavior-data/foraging_nwb_bonsai_processed/foraging_auto_training/'

def copy_folder_on_s3(src_prefix=src_prefix, dst_prefix=dst_prefix):
    # Get a clean list of *files only*
    for src_path in fs.find(src_prefix):
        # 1. Skip placeholder “folders”
        if src_path.endswith('/'):
            continue

        rel_path = os.path.relpath(src_path, src_prefix)

        # 2. Skip the prefix itself (“.”) in case it sneaks in
        if rel_path in ('.', ''):
            continue

        dst_path = f"{dst_prefix.rstrip('/')}/{rel_path}"
        # Fast, server-side copy
        fs.copy(src_path, dst_path)     # or fs.copy_file() in newer fsspec
