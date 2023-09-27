import synapseclient
from dotenv import load_dotenv, dotenv_values

load_dotenv()
syn = synapseclient.login()

entity = syn.get("syn52430419", downloadLocation="/Users/trungle/Desktop/local_map/data/from_synapse")