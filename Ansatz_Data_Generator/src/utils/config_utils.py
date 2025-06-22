#!/usr/bin/env python3
"""
설정 관리를 위한 유틸리티 모듈

딕셔너리에 속성 접근 방식(.)을 사용할 수 있도록 하는 ConfigBox 클래스를 제공합니다.
"""

from typing import Dict, Any, List, Optional, Union


class ConfigBox:
    """
    속성 접근 방식으로 딕셔너리 값에 접근할 수 있는 클래스
    
    예시:
        config_box = ConfigBox({'section': {'key': 'value'}})
        value = config_box.section.key  # 'value' 반환
    """
    
    def __init__(self, dictionary: Optional[Dict[str, Any]] = None):
        """
        딕셔너리로 ConfigBox 초기화
        
        Args:
            dictionary: 초기화할 딕셔너리
        """
        self._data = {}
        
        if dictionary is not None:
            for key, value in dictionary.items():
                # 중첩된 딕셔너리도 ConfigBox로 변환
                if isinstance(value, dict):
                    self._data[key] = ConfigBox(value)
                # 리스트 내의 딕셔너리도 ConfigBox로 변환
                elif isinstance(value, list):
                    self._data[key] = [
                        ConfigBox(item) if isinstance(item, dict) else item 
                        for item in value
                    ]
                else:
                    self._data[key] = value
    
    def __getattr__(self, key: str) -> Any:
        """
        속성 접근 방식으로 딕셔너리 값을 가져옴
        
        Args:
            key: 가져올 키
        
        Returns:
            키에 해당하는 값
        
        Raises:
            AttributeError: 키가 존재하지 않을 경우
        """
        if key in self._data:
            return self._data[key]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key: str, value: Any):
        """
        속성 접근 방식으로 딕셔너리 값을 설정
        
        Args:
            key: 설정할 키
            value: 설정할 값
        """
        if key == '_data':
            super().__setattr__(key, value)
        else:
            # 값이 딕셔너리면 ConfigBox로 변환
            if isinstance(value, dict):
                self._data[key] = ConfigBox(value)
            # 리스트 내의 딕셔너리도 ConfigBox로 변환
            elif isinstance(value, list):
                self._data[key] = [
                    ConfigBox(item) if isinstance(item, dict) else item 
                    for item in value
                ]
            else:
                self._data[key] = value
    
    def __getitem__(self, key: str) -> Any:
        """
        딕셔너리 스타일로 값에 접근
        
        Args:
            key: 가져올 키
            
        Returns:
            키에 해당하는 값
        """
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        새 값 설정
        
        Args:
            key: 설정할 키
            value: 설정할 값
        """
        # 값이 딕셔너리면 ConfigBox로 변환
        if isinstance(value, dict):
            self._data[key] = ConfigBox(value)
        # 리스트 내의 딕셔너리도 ConfigBox로 변환
        elif isinstance(value, list):
            self._data[key] = [
                ConfigBox(item) if isinstance(item, dict) else item 
                for item in value
            ]
        else:
            self._data[key] = value
    
    def __contains__(self, key: str) -> bool:
        """
        키가 존재하는지 확인
        
        Args:
            key: 확인할 키
        
        Returns:
            키가 존재하면 True, 아니면 False
        """
        return key in self._data
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        키에 해당하는 값을 가져오되, 키가 없으면 기본값 반환
        
        Args:
            key: 가져올 키
            default: 키가 없을 경우 반환할 기본값
        
        Returns:
            키에 해당하는 값 또는 기본값
        """
        return self._data.get(key, default)
    
    def update(self, dictionary: Dict):
        """
        다른 딕셔너리로 업데이트
        
        Args:
            dictionary: 업데이트할 딕셔너리
        """
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # 키가 이미 존재하고 ConfigBox라면 재귀적으로 업데이트
                if key in self._data and isinstance(self._data[key], ConfigBox):
                    self._data[key].update(value)
                else:
                    # 아니라면 새로운 ConfigBox 생성
                    self._data[key] = ConfigBox(value)
            else:
                self._data[key] = value

    def items(self):
        """
        딕셔너리의 items() 메서드 구현
        
        Returns:
            (key, value) 튜플들의 iterator
        """
        return self._data.items()
    
    def to_dict(self) -> Dict:
        """
        ConfigBox를 일반 딕셔너리로 변환
        
        Returns:
            일반 딕셔너리
        """
        result = {}
        for key, value in self._data.items():
            if isinstance(value, ConfigBox):
                # ConfigBox는 재귀적으로 변환
                result[key] = value.to_dict()
            elif isinstance(value, list):
                # 리스트 내의 ConfigBox도 재귀적으로 변환
                result[key] = [
                    item.to_dict() if isinstance(item, ConfigBox) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result
    
    def __repr__(self) -> str:
        """
        ConfigBox를 문자열로 표현
        
        Returns:
            문자열 표현
        """
        return f"{self.__class__.__name__}({self._data})"
    

