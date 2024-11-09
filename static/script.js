function toggleSidebar() {
    const $sidebar = $('#sidebar');
    const $toggleButton = $('.toggle-button');
    
    $sidebar.toggleClass('active');
    $toggleButton.attr('title', $sidebar.hasClass('active') ? 'Close Menu' : 'Open Menu');
}

$(document).ready(function() {
    const $toggleButton = $('.toggle-button');
    const $sidebar = $('#sidebar');

    $toggleButton.attr('title', $sidebar.hasClass('active') ? 'Close Menu' : 'Open Menu');
});

function resizeSidebar() {
    const $sidebar = $('.sidebar');
    const $toggleButton = $('.sidebar-toggle-button');

    if ($sidebar.css('width') === '250px' || $sidebar.css('width') === 'auto') {
        $sidebar.css('width', '80px');
        $toggleButton.html('&#62;&#62;').attr('title', 'Open sidebar');
    } else {
        $sidebar.css('width', '250px');
        $toggleButton.html('&#60;&#60;').attr('title', 'Close sidebar');
    }
}

$(document).ready(function() {
    const $toggleButton = $('.sidebar-toggle-button');
    $toggleButton.attr('title', 'Close sidebar');
});

const $userInput = $('#user-query');
const $sendButton = $('#send-button');

$userInput.on('input', function() {
    if ($userInput.val().trim() !== "") {
        $sendButton.addClass('active').removeClass('disabled').prop('disabled', false);
    } else {
        $sendButton.removeClass('active').addClass('disabled').prop('disabled', true);
    }
});

let isLoading = false;

$('#send-button').on('click', sendMessage);

$('#user-query').on('keydown', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        sendMessage();
    }
});

function typeMessage($element, message) {
    const words = message.split(" ");
    let index = 0;

    const interval = setInterval(() => {
        if (index < words.length) {
            $element.append(words[index] + " ");
            index++;
        } else {
            clearInterval(interval);
        }
    }, 100);
}

function sendMessage() {
    const query = $userInput.val().trim();
    if (!query || isLoading) return;

    $sendButton.prop('disabled', true).removeClass('active').addClass('disabled');

    isLoading = true;
    $('#loading-indicator').text("Loading...");

    const $chatOutput = $('#chat-output');
    $chatOutput.append(`
        <div class="chat-message user">
            <div class="avatar user-avatar" style="background-image: url('https://media.istockphoto.com/id/1300845620/vector/user-icon-flat-isolated-on-white-background-user-symbol-vector-illustration.jpg?s=612x612&w=0&k=20&c=yBeyba0hUkh14_jgv1OKqIH0CCSWU_4ckRkAoy2p73o=');"></div>
            <div class="message">${query}</div>
        </div>
    `);

    $userInput.val('');
    $chatOutput.scrollTop($chatOutput.prop('scrollHeight'));

    const $typingIndicator = $(`
        <div class="chat-message bot typing-indicator">
            <div class="avatar bot-avatar" style="background-image: url('https://media.istockphoto.com/id/2148171304/vector/chatbot-avatar.jpg?s=612x612&w=0&k=20&c=ipIX_zqUkcRsYyI_CSO5eIDm_sjtbCwWVMVWRVp07PE=');"></div>
            <div class="message"><span>.</span><span>.</span><span>.</span></div>
        </div>
    `);
    $chatOutput.append($typingIndicator);
    $chatOutput.scrollTop($chatOutput.prop('scrollHeight'));

    $.ajax({
        url: 'http://127.0.0.1:5000/chatbot',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ query: query }),
        success: function(data) {
            setTimeout(() => {
                $typingIndicator.remove();

                if (Array.isArray(data.answer)) {
                    data.answer.forEach(answer => {
                        const $botMessage = $(`
                            <div class="chat-message bot">
                                <div class="avatar bot-avatar" style="background-image: url('https://media.istockphoto.com/id/2148171304/vector/chatbot-avatar.jpg?s=612x612&w=0&k=20&c=ipIX_zqUkcRsYyI_CSO5eIDm_sjtbCwWVMVWRVp07PE=');"></div>
                                <div class="message"></div>
                            </div>
                        `);
                        $chatOutput.append($botMessage);
                        typeMessage($botMessage.find('.message'), answer);
                    });
                } else {
                    const $botMessage = $(`
                        <div class="chat-message bot">
                            <div class="avatar bot-avatar" style="background-image: url('https://media.istockphoto.com/id/2148171304/vector/chatbot-avatar.jpg?s=612x612&w=0&k=20&c=ipIX_zqUkcRsYyI_CSO5eIDm_sjtbCwWVMVWRVp07PE=');"></div>
                            <div class="message"></div>
                        </div>
                    `);
                    $chatOutput.append($botMessage);
                    typeMessage($botMessage.find('.message'), data.answer);
                }

                $chatOutput.scrollTop($chatOutput.prop('scrollHeight'));
                isLoading = false;
                $('#loading-indicator').text("");
            }, 800);
        }
    });
}
