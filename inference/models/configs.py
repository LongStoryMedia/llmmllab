from sympy import true
import server.config as config
from .rabbitmq_config import RabbitmqConfig


def rabbitmq_config() -> RabbitmqConfig:
    """
    Returns the RabbitMQ configuration as a dictionary.
    This is used to configure the RabbitMQ consumer service.
    """
    return RabbitmqConfig(
        host=config.RABBITMQ_HOST,
        port=int(config.RABBITMQ_PORT),
        user=config.RABBITMQ_USER,
        password=config.RABBITMQ_PASSWORD,
        vhost=config.RABBITMQ_VHOST,
        enabled=True,
    )
