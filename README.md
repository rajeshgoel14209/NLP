1. Use Environment Variables Instead of Hardcoded Values

	import os

	DB_HOST = os.getenv("DB_HOST", "localhost")
	DB_USER = os.getenv("DB_USER", "default_user")
	DB_PASSWORD = os.getenv("DB_PASSWORD", "default_pass")

2. Avoids exposing sensitive credentials in code
3. Enables different configs per environment (SIT/UAT/PROD
	for example:
		common_config_DEV.json
		common_config_SIT.json
		common_config_PROD.json


4. Use Proper Logging & Monitoring Instead of print() for Debugging 

	import logging

	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
	logging.info("Application started successfully")

5. Differentiate logs by environment (sit-logs, uat-logs, prod-logs)

6. Use consistent naming conventions, indentation, and docstrings.
7. Explicitly define function parameter and return types.

8.  Catch specific exceptions and avoid generic except Exception
9.  Use list comprehensions and generators for better performance.
10. Use FastAPI health checks for ECS/Kubernetes.
	@app.get("/health")
	def health_check():
		return {"status": "ok"}
