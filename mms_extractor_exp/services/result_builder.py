"""
Result Builder Service
=======================

최종 추출 결과를 조립하는 서비스 클래스입니다.
채널 정보 추출, 프로그램 분류 매핑, offer 객체 생성을 담당합니다.

엔티티 매칭 로직은 EntityMatchingStep으로 분리되었습니다.
"""

import logging
import pandas as pd
import re
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ResultBuilder:
    """
    Service for assembling the final extraction result.
    Handles channel extraction, program mapping, offer creation, and result assembly.

    Entity matching is handled separately by EntityMatchingStep.
    """

    def __init__(self, store_matcher, stop_item_names: List[str], num_cand_pgms: int):
        self.store_matcher = store_matcher
        self.stop_item_names = stop_item_names
        self.num_cand_pgms = num_cand_pgms

    def assemble_result(self, json_objects: Dict, matched_products: List[Dict],
                        msg: str, pgm_info: Dict, message_id: str = '#') -> Dict[str, Any]:
        """
        최종 결과 조립

        Args:
            json_objects: LLM 파싱 결과
            matched_products: EntityMatchingStep에서 매칭된 상품 목록
            msg: 원본 메시지
            pgm_info: 프로그램 분류 정보
            message_id: 메시지 ID

        Returns:
            최종 결과 딕셔너리
        """
        try:
            final_result = json_objects.copy()

            # 상품 정보 설정 (EntityMatchingStep에서 매칭 완료)
            final_result['product'] = matched_products
            logger.info(f"상품 수: {len(matched_products)}개")

            # offer_object 초기화 (product 타입)
            offer_object = {
                'type': 'product',
                'value': matched_products
            }

            # 프로그램 분류 정보 매핑
            final_result['pgm'] = self._map_programs_to_result(json_objects, pgm_info)

            # 채널 정보 처리 (offer_object도 함께 전달 및 반환)
            final_result['channel'], offer_object = self._extract_and_enrich_channels(
                json_objects, msg, offer_object
            )
            logger.info(f"채널 수: {len(final_result['channel'])}개, offer type: {offer_object.get('type')}")

            # offer 필드 추가
            final_result['offer'] = offer_object

            # entity_dag 초기화 (빈 배열, DAGExtractionStep에서 설정)
            final_result['entity_dag'] = []

            # message_id 추가
            final_result['message_id'] = message_id

            return final_result

        except Exception as e:
            logger.error(f"최종 결과 구성 실패: {e}")
            return json_objects

    # Backward compatibility alias
    def build_extraction_result(self, json_objects: Dict, msg: str, pgm_info: Dict,
                                entities_from_kiwi: List[str], message_id: str = '#') -> Dict[str, Any]:
        """Deprecated: use assemble_result() instead. Kept for backward compatibility."""
        logger.warning("build_extraction_result() is deprecated, use assemble_result()")
        # When called via old interface, matched_products defaults to empty
        return self.assemble_result(json_objects, [], msg, pgm_info, message_id)

    def _map_programs_to_result(self, json_objects: Dict, pgm_info: Dict) -> List[Dict]:
        """프로그램 분류 정보 매핑"""
        try:
            if (self.num_cand_pgms > 0 and
                'pgm' in json_objects and
                isinstance(json_objects['pgm'], list) and
                not pgm_info.get('pgm_pdf_tmp', pd.DataFrame()).empty):

                pgm_json = pgm_info['pgm_pdf_tmp'][
                    pgm_info['pgm_pdf_tmp']['pgm_nm'].apply(
                        lambda x: re.sub(r'\[.*?\]', '', x) in ' '.join(json_objects['pgm'])
                    )
                ][['pgm_nm', 'pgm_id']].to_dict('records')

                return pgm_json

            return []

        except Exception as e:
            logger.error(f"프로그램 분류 매핑 실패: {e}")
            return []

    def _extract_and_enrich_channels(self, json_objects: Dict, msg: str, offer_object: Dict) -> Tuple[List[Dict], Dict]:
        """채널 정보 추출 및 매칭 (offer_object도 함께 반환)"""
        try:
            channel_tag = []
            channel_items = json_objects.get('channel', [])
            if isinstance(channel_items, dict):
                channel_items = channel_items.get('items', [])

            for d in channel_items:
                if d.get('type') == '대리점' and d.get('value'):
                    # 대리점명으로 조직 정보 검색
                    store_info = self.store_matcher.match_store(d['value'])
                    d['store_info'] = store_info

                    # offer_object를 org 타입으로 변경
                    if store_info:
                        offer_object['type'] = 'org'
                        org_tmp = [
                            {
                                'item_nm': o['org_nm'],
                                'item_id': o['org_cd'],
                                'item_name_in_msg': d['value'],
                                'expected_action': ['방문']
                            }
                            for o in store_info
                        ]
                        offer_object['value'] = org_tmp
                    else:
                        if "대리점/매장 방문 유도" in json_objects['purpose']:
                            offer_object['type'] = 'org'
                            org_tmp = [{'item_nm':d['value'], 'item_id':'#', 'item_name_in_msg':d['value'], 'expected_action':['방문']}]
                            offer_object['value'] = org_tmp
                else:
                    d['store_info'] = []
                channel_tag.append(d)

            return channel_tag, offer_object

        except Exception as e:
            logger.error(f"채널 정보 추출 실패: {e}")
            return [], offer_object
