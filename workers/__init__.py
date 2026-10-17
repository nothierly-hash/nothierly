"""Workers de plateformes : un process isolé par plateforme, alimenté par le bus ZeroMQ signé.

Frontière d'IP : un worker n'importe que la stdlib, ``zmq``, ``contracts`` et le SDK de sa
plateforme. Jamais ``core`` ni ``src`` (code du moteur) ; ``tests/test_numerai_worker.py`` le vérifie.
"""
