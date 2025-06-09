import s3fs
import os

# Initialize the S3 filesystem
fs = s3fs.S3FileSystem(anon=False)  # anon=False for authenticated access

src_prefix = 'aind-behavior-data/foraging_auto_training/'
dst_prefix = 'aind-behavior-data/foraging_nwb_bonsai_processed/foraging_auto_training/'

def copy_folder_on_s3(src_prefix=src_prefix, dst_prefix=dst_prefix):

    # List all files under the source prefix
    files = fs.find(src_prefix)  # Recursively lists all files

    for src_path in files:
        # Get the relative path after the src_prefix
        rel_path = os.path.relpath(src_path, src_prefix)
        dst_path = os.path.join(dst_prefix, rel_path)
        
        # Make sure the destination directory exists (S3 "folders" are implicit)
        # Copy file from src_path to dst_path
        with fs.open(src_path, 'rb') as fsrc:
            with fs.open(dst_path, 'wb') as fdst:
                fdst.write(fsrc.read())
        print(f"Copied {src_path} to {dst_path}")
