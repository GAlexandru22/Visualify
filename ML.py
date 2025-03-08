import google
from google import genai
from typing import Optional
import vertexai

client = genai.Client(api_key="AIzaSyCsdb9bxS4JgO6PRDC2gHDe5NkhmB2fANs") #API-Key
response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain how AI works"
)
print(response.text)

def init_sample(
    project: Optional[str] = None,
    location: Optional[str] = None,
    experiment: Optional[str] = None,
    staging_bucket: Optional[str] = None,
    credentials: Optional[google.auth.credentials.Credentials] = None,
    encryption_spec_key_name: Optional[str] = None,
    service_account: Optional[str] = None,
):

    vertexai.init(
        project=project,
        location=location,
        experiment=experiment,
        staging_bucket=staging_bucket,
        credentials=credentials,
        encryption_spec_key_name=encryption_spec_key_name,
        service_account=service_account,
    )