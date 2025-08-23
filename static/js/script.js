document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.star-rating').forEach(star => {
        star.addEventListener('click', function() {
            const rating = this.getAttribute('data-rating');
            const mediaTitle = this.getAttribute('data-media-title');
            const media_id = this.getAttribute('data-media-id');
            
            fetch('/rate_media', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    media_id: media_id,
                    rating: rating,
                    media_title: mediaTitle,
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const stars = document.querySelectorAll(`.star-rating[data-media-id="${media_id}"]`);
                    stars.forEach((star, index) => {
                        star.classList.toggle('text-yellow-400', index < rating);
                        star.classList.toggle('text-gray-500', index >= rating);
                    });
                    
                    const rating_display = document.querySelector('.rating-display');
                    if (rating_display) {
                        rating_display.textContent = `(${rating}/5)`;
                    }
                } else {
                    console.error('Error:', data.error);
                }
            })
            .catch(error => console.error('Error:', error));
        });
    });
});

document.addEventListener('DOMContentLoaded', function() {
    const delete_form = document.querySelector('form[action="/delete-all-ratings"]');
    if (delete_form) {
        delete_form.addEventListener('submit', function(e) {
            if (!confirm('Are you sure you want to delete ALL your ratings? This cannot be undone.')) {
                e.preventDefault();
            }
        });
    }
});