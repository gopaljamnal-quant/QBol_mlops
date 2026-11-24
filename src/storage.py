import io
import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from src.config import settings

def get_blob_service_client() -> BlobServiceClient:
    credential = DefaultAzureCredential()
    return BlobServiceClient(account_url=settings.AZURE_STORAGE_ACCOUNT_URL, credential=credential)

def read_csv_from_blob(blob_path: str) -> pd.DataFrame:
    client = get_blob_service_client()
    blob_client = client.get_blob_client(container=settings.AZURE_CONTAINER_NAME, blob=blob_path)
    data = blob_client.download_blob().readall()
    return pd.read_csv(io.BytesIO(data))

def write_csv_to_blob(df: pd.DataFrame, blob_path: str) -> None:
    client = get_blob_service_client()
    container_client = client.get_container_client(settings.AZURE_CONTAINER_NAME)
    if not container_client.exists():
        container_client.create_container()
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    blob_client = container_client.get_blob_client(blob=blob_path)
    blob_client.upload_blob(csv_buffer.getvalue(), overwrite=True)

def append_inference_log(new_row_df: pd.DataFrame) -> None:
    """Appends live predictions to Azure Blob Storage for Evidently drift monitoring."""
    try:
        existing_df = read_csv_from_blob(settings.INFERENCE_LOGS_BLOB)
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    except Exception:
        updated_df = new_row_df
    write_csv_to_blob(updated_df, settings.INFERENCE_LOGS_BLOB)
