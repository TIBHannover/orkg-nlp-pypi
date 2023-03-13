# -*- coding: utf-8 -*-
from typing import Any

from orkgnlp.common.config import orkgnlp_context
from orkgnlp.common.service.base import ORKGNLPBaseService
from orkgnlp.common.util.exceptions import ORKGNLPUnknownServiceError


def load(service_name: str, *args: Any, **kwargs: Any) -> ORKGNLPBaseService:
    """
    Creates a class object of the requested service.

    :param service_name: The name of the service class.
    :return: A service class object.
    """
    try:
        return orkgnlp_context["SERVICE_MAP"][service_name](*args, **kwargs)
    except KeyError:
        raise ORKGNLPUnknownServiceError(
            "{} is not known.  Please use one of: {}".format(
                service_name, list(orkgnlp_context["SERVICE_MAP"].keys())
            )
        )
