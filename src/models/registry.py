from src.models.providers import cloud_api
def get_model(config: dict):
    provider = config["provider"]
    model_name = config["model_name"]

    if provider == "cloud":
        # timeout/max_retries — опциональные настройки, чтобы вызовы не могли висеть бесконечно
        timeout = config.get("timeout") or config.get("request_timeout")
        max_retries = config.get("max_retries")
        return cloud_api.CloudAPIModel(
            model_name=model_name,
            temperature=config.get("temperature", 0.7),
            timeout=timeout,
            max_retries=max_retries,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
