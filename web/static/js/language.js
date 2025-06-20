document.addEventListener('DOMContentLoaded', function() {
    // Khởi tạo ngôn ngữ từ localStorage hoặc mặc định
    let currentLanguage = localStorage.getItem('preferredLanguage') || 
                          (document.documentElement.lang === 'vi' ? 'vi' : 'en');
    
    // Áp dụng ngôn ngữ khi tải trang
    applyLanguage(currentLanguage);
    
    // Nút chuyển đổi ngôn ngữ
    const languageToggle = document.getElementById('language-toggle');
    if (languageToggle) {
        // Cập nhật hiển thị nút
        updateLanguageButton(currentLanguage);
        
        // Thêm sự kiện click
        languageToggle.addEventListener('click', function() {
            // Đổi ngôn ngữ
            currentLanguage = currentLanguage === 'en' ? 'vi' : 'en';
            
            // Lưu vào localStorage
            localStorage.setItem('preferredLanguage', currentLanguage);
            
            // Áp dụng ngôn ngữ mới
            applyLanguage(currentLanguage);
            
            // Cập nhật hiển thị nút
            updateLanguageButton(currentLanguage);
        });
    }
    
    function applyLanguage(lang) {
        // Cập nhật thuộc tính lang của html
        document.documentElement.lang = lang;
        
        // Cập nhật tiêu đề trang
        const titleElement = document.querySelector('title');
        if (titleElement && titleElement.getAttribute('data-' + lang)) {
            titleElement.textContent = titleElement.getAttribute('data-' + lang);
        }
        
        // Chuyển đổi tất cả phần tử có thuộc tính data-en và data-vi
        const elements = document.querySelectorAll('[data-' + lang + ']');
        elements.forEach(element => {
            element.textContent = element.getAttribute('data-' + lang);
        });
        
        // Cập nhật placeholder cho các trường input
        const inputs = document.querySelectorAll('[data-' + lang + '-placeholder]');
        inputs.forEach(input => {
            input.placeholder = input.getAttribute('data-' + lang + '-placeholder');
        });
    }
    
    function updateLanguageButton(lang) {
        const langDisplay = languageToggle.querySelector('.current-lang');
        if (langDisplay) {
            langDisplay.textContent = lang.toUpperCase();
        }
    }
});