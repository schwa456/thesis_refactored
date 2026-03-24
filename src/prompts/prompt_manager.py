import os
import re
from utils.logger import get_logger

logger = get_logger(__name__)

class PromptManager:
    def __init__(self, prompt_dir: str = None):
        if prompt_dir is None:
            self.prompt_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.prompt_dir = prompt_dir

        self._cache = {}
    
    def load_prompt(self, file_name: str, section: str = None, **kwargs) -> str:
        """
        마크다운 파일을 읽어옵니다. section이 주어지면 해당 '## section' 하위의 텍스트만 추출합니다.
        """
        if not file_name.endswith('.md'):
            file_name += '.md'
        
        file_path = os.path.join(self.prompt_dir, file_name)

        # 1. 파일 캐싱
        if file_name not in self._cache:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"🚨 Prompt file not found: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                self._cache[file_name] = f.read()
                
        content = self._cache[file_name]
        template = content

        # 2. Section 파싱 로직 (특정 프롬프트만 잘라내기)
        if section:
            # '## section이름' 으로 시작해서 다음 '##'이 나오거나 파일이 끝날 때까지 매칭
            pattern = rf"^##\s+{re.escape(section)}\s*\n(.*?)(?=^##\s+|\Z)"
            match = re.search(pattern, content, flags=re.MULTILINE | re.DOTALL)
            
            if not match:
                raise ValueError(f"🚨 Section '{section}' not found in {file_name}")
            
            template = match.group(1).strip()

        # 3. 변수 주입(Formatting)
        try:
            return template.format(**kwargs) if kwargs else template
        except KeyError as e:
            logger.error(f"Missing variable {e} in prompt '{file_name}' (section: {section})")
            raise