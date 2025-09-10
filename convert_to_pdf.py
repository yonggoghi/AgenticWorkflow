#!/usr/bin/env python3
"""
Markdown to PDF 변환 스크립트
MMS Extractor 프레젠테이션을 PDF로 변환합니다.
"""

import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
import os

def convert_markdown_to_pdf(markdown_file, output_file):
    """Markdown 파일을 PDF로 변환"""
    
    # Markdown 파일 읽기
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Markdown을 HTML로 변환
    html_content = markdown.markdown(
        markdown_content,
        extensions=['tables', 'fenced_code', 'toc']
    )
    
    # CSS 스타일 정의 (프레젠테이션용)
    css_style = """
    @page {
        size: A4;
        margin: 2cm;
    }
    
    body {
        font-family: 'Helvetica', 'Arial', sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: none;
    }
    
    h1 {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
        margin-top: 40px;
        margin-bottom: 30px;
        font-size: 28px;
        page-break-before: always;
    }
    
    h1:first-of-type {
        page-break-before: avoid;
        text-align: center;
        margin-top: 0;
        font-size: 36px;
        color: #e74c3c;
    }
    
    h2 {
        color: #34495e;
        margin-top: 35px;
        margin-bottom: 20px;
        font-size: 24px;
        border-left: 5px solid #3498db;
        padding-left: 15px;
    }
    
    h3 {
        color: #2980b9;
        margin-top: 25px;
        margin-bottom: 15px;
        font-size: 20px;
    }
    
    h4 {
        color: #8e44ad;
        margin-top: 20px;
        margin-bottom: 10px;
        font-size: 18px;
    }
    
    p {
        margin-bottom: 15px;
        text-align: justify;
    }
    
    ul, ol {
        margin-bottom: 20px;
        padding-left: 25px;
    }
    
    li {
        margin-bottom: 8px;
    }
    
    code {
        background-color: #f8f9fa;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Monaco', 'Consolas', monospace;
        font-size: 14px;
    }
    
    pre {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #3498db;
        overflow-x: auto;
        margin: 20px 0;
    }
    
    pre code {
        background: none;
        padding: 0;
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
        font-size: 14px;
    }
    
    th, td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
    }
    
    th {
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
    
    tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    
    blockquote {
        border-left: 4px solid #3498db;
        margin: 20px 0;
        padding: 10px 20px;
        background-color: #f8f9fa;
        font-style: italic;
    }
    
    .emoji {
        font-size: 1.2em;
    }
    
    hr {
        border: none;
        border-top: 2px solid #3498db;
        margin: 40px 0;
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
    
    /* 페이지 구분선 스타일 */
    .page-break {
        page-break-before: always;
    }
    
    /* 목차 스타일 */
    .toc {
        background-color: #ecf0f1;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
    }
    
    /* 코드 블록 언어별 스타일 */
    .language-bash {
        border-left-color: #27ae60;
    }
    
    .language-json {
        border-left-color: #f39c12;
    }
    
    .language-python {
        border-left-color: #3776ab;
    }
    """
    
    # HTML 템플릿 생성
    html_template = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="utf-8">
        <title>MMS Extractor - 과제 소개 자료</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # PDF 생성
    print(f"Converting {markdown_file} to PDF...")
    
    font_config = FontConfiguration()
    html_doc = HTML(string=html_template)
    css_doc = CSS(string=css_style, font_config=font_config)
    
    html_doc.write_pdf(
        output_file,
        stylesheets=[css_doc],
        font_config=font_config
    )
    
    print(f"PDF 생성 완료: {output_file}")
    
    # 파일 크기 확인
    file_size = os.path.getsize(output_file)
    print(f"파일 크기: {file_size / 1024 / 1024:.2f} MB")

def main():
    """메인 함수"""
    markdown_file = "MMS_Extractor_Presentation.md"
    output_file = "MMS_Extractor_과제소개자료.pdf"
    
    if not os.path.exists(markdown_file):
        print(f"오류: {markdown_file} 파일을 찾을 수 없습니다.")
        return
    
    try:
        convert_markdown_to_pdf(markdown_file, output_file)
        print("\n✅ PDF 변환이 성공적으로 완료되었습니다!")
        print(f"📄 생성된 파일: {output_file}")
        
    except Exception as e:
        print(f"❌ PDF 변환 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
