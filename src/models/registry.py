from src.models.providers import cloud_api, openrouter_api


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
    elif provider == "openrouter":
        # OpenRouter API с rate limiting
        timeout = config.get("timeout")
        max_retries = config.get("max_retries")
        max_completion_tokens = config.get("max_completion_tokens", 4000)
        extra_body = config.get("extra_body")

        # Rate limiting настройки
        rate_limit_config = config.get("rate_limit", {})
        rate_limit_enabled = rate_limit_config.get("enabled", True)
        requests_per_second = rate_limit_config.get("requests_per_second", 1.0)
        max_bucket_size = rate_limit_config.get("max_bucket_size", 10)

        return openrouter_api.OpenRouterAPIModel(
            model_name=model_name,
            temperature=config.get("temperature", 0.7),
            timeout=timeout,
            max_retries=max_retries,
            max_completion_tokens=max_completion_tokens,
            extra_body=extra_body,
            rate_limit_enabled=rate_limit_enabled,
            requests_per_second=requests_per_second,
            max_bucket_size=max_bucket_size,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
