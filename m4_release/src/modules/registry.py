import logging

logger = logging.getLogger(__name__)

# ==========================================
# 🗄️ 글로벌 레지스트리 저장소
# ==========================================
REGISTRY = {
    "builder": {},
    "encoder": {},
    "projector": {},
    "selector": {},
    "extractor": {},
    "filter": {},
    "generator": {}
}

def register(category: str, name: str):
    """
    모듈을 레지스트리에 등록하는 데코레이터
    """

    def decorator(cls):
        if category not in REGISTRY:
            REGISTRY[category] = {}
            logger.warning(f"New Category '{category}' created in registry.")

        if name in REGISTRY[category]:
            raise ValueError(f"Module '{name}' is already registered in category '{category}'.")
        
        REGISTRY[category][name] = cls
        logger.debug(f"✅ Registered [{category}] -> {name}")
        return cls

    return decorator

def build(category: str, config: dict, **kwargs):
    """
    Config 설정값을 바탕으로 등록된 클래스의 객체를 생성(Instantiate)하여 반환
    """

    if category not in REGISTRY:
        raise ValueError(f"🚨 Category '{category}' is not supported in the registry.")

    if not config or 'name' not in config:
        raise ValueError(f"🚨 Config for '{category}' must contain a 'name' key. Config given: {config}")
    
    name = config['name']
    if name not in REGISTRY[category]:
        available = list(REGISTRY[category].keys())
        raise ValueError(f"🚨 Module '{name}' not found in category '{category}'. Available modules: {available}")
    
    target_class = REGISTRY[category][name]

    init_params = config.get('params', {}).copy()
    init_params.update(kwargs)

    logger.debug(f"🔨 Building [{category}] -> {name} | Params: {init_params}")

    return target_class(**init_params)