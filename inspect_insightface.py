import insightface
from insightface.app import FaceAnalysis
import inspect

# Print the docstring for the FaceAnalysis class
print("--- FaceAnalysis Docstring ---")
print(FaceAnalysis.__doc__)
print("\n" + "="*50 + "\n")

# Get the signature of the 'get' method
try:
    get_method = getattr(FaceAnalysis, 'get')
    print("--- FaceAnalysis.get Signature ---")
    print(inspect.signature(get_method))
    print("\n" + "="*50 + "\n")

    # Print the docstring for the 'get' method
    print("--- FaceAnalysis.get Docstring ---")
    print(get_method.__doc__)
except AttributeError:
    print("Could not find the 'get' method on the FaceAnalysis class.")
