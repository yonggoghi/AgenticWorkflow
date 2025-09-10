#!/usr/bin/env python3
"""
Markdown을 HTML 프레젠테이션으로 변환
브라우저에서 PDF로 인쇄할 수 있도록 최적화된 HTML 생성
"""

import markdown
import os

def convert_markdown_to_html(markdown_file, output_file):
    """Markdown 파일을 프레젠테이션용 HTML로 변환"""
    
    # Markdown 파일 읽기
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Markdown을 HTML로 변환
    html_content = markdown.markdown(
        markdown_content,
        extensions=['tables', 'fenced_code', 'toc']
    )
    
    # CSS 스타일 정의 (프레젠테이션 및 인쇄용 최적화)
    css_style = """
    <style>
    /* 인쇄용 스타일 */
    @media print {
        @page {
            size: A4;
            margin: 2cm;
        }
        
        body {
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
        }
        
        h1 {
            page-break-before: always;
        }
        
        h1:first-of-type {
            page-break-before: avoid;
        }
        
        h2, h3, h4 {
            page-break-after: avoid;
        }
        
        table, pre, blockquote {
            page-break-inside: avoid;
        }
    }
    
    /* 기본 스타일 */
    body {
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 1000px;
        margin: 0 auto;
        padding: 20px;
        background-color: #fff;
    }
    
    /* 제목 스타일 */
    h1 {
        color: #2c3e50;
        border-bottom: 4px solid #3498db;
        padding-bottom: 15px;
        margin-top: 50px;
        margin-bottom: 30px;
        font-size: 32px;
        font-weight: bold;
    }
    
    h1:first-of-type {
        text-align: center;
        margin-top: 0;
        font-size: 42px;
        color: #e74c3c;
        border-bottom: none;
        padding-bottom: 0;
    }
    
    h2 {
        color: #34495e;
        margin-top: 40px;
        margin-bottom: 20px;
        font-size: 28px;
        border-left: 6px solid #3498db;
        padding-left: 20px;
        background: linear-gradient(90deg, #ecf0f1 0%, transparent 100%);
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    h3 {
        color: #2980b9;
        margin-top: 30px;
        margin-bottom: 15px;
        font-size: 24px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
    }
    
    h4 {
        color: #8e44ad;
        margin-top: 25px;
        margin-bottom: 12px;
        font-size: 20px;
        font-weight: 600;
    }
    
    /* 텍스트 스타일 */
    p {
        margin-bottom: 16px;
        text-align: justify;
    }
    
    ul, ol {
        margin-bottom: 20px;
        padding-left: 30px;
    }
    
    li {
        margin-bottom: 8px;
    }
    
    /* 강조 표시 */
    strong {
        color: #e74c3c;
        font-weight: bold;
    }
    
    em {
        color: #8e44ad;
        font-style: italic;
    }
    
    /* 코드 스타일 */
    code {
        background-color: #f8f9fa;
        padding: 3px 8px;
        border-radius: 4px;
        font-family: 'Monaco', 'Consolas', 'Courier New', monospace;
        font-size: 14px;
        color: #d73a49;
        border: 1px solid #e1e4e8;
    }
    
    pre {
        background-color: #f6f8fa;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #3498db;
        overflow-x: auto;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    pre code {
        background: none;
        padding: 0;
        border: none;
        color: #24292e;
    }
    
    /* 테이블 스타일 */
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 25px 0;
        font-size: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-radius: 8px;
        overflow: hidden;
    }
    
    th, td {
        border: 1px solid #ddd;
        padding: 15px;
        text-align: left;
    }
    
    th {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    
    tr:hover {
        background-color: #e3f2fd;
    }
    
    /* 인용문 스타일 */
    blockquote {
        border-left: 5px solid #3498db;
        margin: 25px 0;
        padding: 15px 25px;
        background: linear-gradient(90deg, #f8f9fa 0%, #fff 100%);
        font-style: italic;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* 구분선 */
    hr {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, #3498db, #2980b9);
        margin: 50px 0;
        border-radius: 2px;
    }
    
    /* 이모지 크기 조정 */
    .emoji {
        font-size: 1.3em;
        vertical-align: middle;
    }
    
    /* 특별 박스 스타일 */
    .highlight-box {
        background: linear-gradient(135deg, #e8f5e8, #f0f8ff);
        border: 2px solid #27ae60;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* 성과 지표 강조 */
    .metric {
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
        border-left: 5px solid #ff9800;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
        font-weight: bold;
    }
    
    /* 반응형 디자인 */
    @media (max-width: 768px) {
        body {
            padding: 15px;
        }
        
        h1 {
            font-size: 28px;
        }
        
        h2 {
            font-size: 24px;
        }
        
        table {
            font-size: 14px;
        }
        
        th, td {
            padding: 10px;
        }
    }
    </style>
    """
    
    # 제목 추가 처리 (이모지 크기 조정)
    html_content = html_content.replace('🎯', '<span class="emoji">🎯</span>')
    html_content = html_content.replace('📊', '<span class="emoji">📊</span>')
    html_content = html_content.replace('🏆', '<span class="emoji">🏆</span>')
    html_content = html_content.replace('🚀', '<span class="emoji">🚀</span>')
    html_content = html_content.replace('🔬', '<span class="emoji">🔬</span>')
    html_content = html_content.replace('🏗️', '<span class="emoji">🏗️</span>')
    html_content = html_content.replace('🔧', '<span class="emoji">🔧</span>')
    html_content = html_content.replace('📝', '<span class="emoji">📝</span>')
    html_content = html_content.replace('⚡', '<span class="emoji">⚡</span>')
    html_content = html_content.replace('📈', '<span class="emoji">📈</span>')
    html_content = html_content.replace('🖥️', '<span class="emoji">🖥️</span>')
    html_content = html_content.replace('🌐', '<span class="emoji">🌐</span>')
    html_content = html_content.replace('📦', '<span class="emoji">📦</span>')
    html_content = html_content.replace('🎨', '<span class="emoji">🎨</span>')
    html_content = html_content.replace('🤖', '<span class="emoji">🤖</span>')
    html_content = html_content.replace('🛠️', '<span class="emoji">🛠️</span>')
    html_content = html_content.replace('📱', '<span class="emoji">📱</span>')
    html_content = html_content.replace('💼', '<span class="emoji">💼</span>')
    html_content = html_content.replace('🌟', '<span class="emoji">🌟</span>')
    html_content = html_content.replace('💡', '<span class="emoji">💡</span>')
    html_content = html_content.replace('📋', '<span class="emoji">📋</span>')
    
    # HTML 템플릿 생성
    html_template = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MMS Extractor - 과제 소개 자료</title>
        {css_style}
    </head>
    <body>
        {html_content}
        
        <script>
        // 인쇄 최적화를 위한 JavaScript
        window.onload = function() {{
            // 페이지 로드 완료 후 자동으로 인쇄 다이얼로그 표시 (선택사항)
            // window.print();
        }};
        </script>
    </body>
    </html>
    """
    
    # HTML 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"HTML 파일 생성 완료: {output_file}")
    
    # 파일 크기 확인
    file_size = os.path.getsize(output_file)
    print(f"파일 크기: {file_size / 1024:.2f} KB")

def main():
    """메인 함수"""
    markdown_file = "MMS_Extractor_Presentation.md"
    output_file = "MMS_Extractor_과제소개자료.html"
    
    if not os.path.exists(markdown_file):
        print(f"오류: {markdown_file} 파일을 찾을 수 없습니다.")
        return
    
    try:
        convert_markdown_to_html(markdown_file, output_file)
        print("\n✅ HTML 변환이 성공적으로 완료되었습니다!")
        print(f"📄 생성된 파일: {output_file}")
        print("\n📖 사용 방법:")
        print("1. 생성된 HTML 파일을 브라우저에서 열기")
        print("2. 브라우저에서 Ctrl+P (또는 Cmd+P) 눌러 인쇄")
        print("3. '대상'을 'PDF로 저장'으로 선택")
        print("4. 여백을 '최소'로 설정하고 '배경 그래픽' 체크")
        print("5. '저장' 버튼 클릭하여 PDF 생성")
        
    except Exception as e:
        print(f"❌ HTML 변환 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
