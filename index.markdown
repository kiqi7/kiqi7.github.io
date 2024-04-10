---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

<!-- ![Qi Chen](assets/img/IMG_2789.PNG) -->
<img style="float:right" src="/assets/img/IMG_2801.PNG" width="369"/>

# Welcome to My Portfolio


<!-- Audio Controls -->
<div id="audio-controls">
  <button onclick="switchLanguage('english')">English</button>
  <!-- <button onclick="switchLanguage('spanish')">Español</button> -->
  <button onclick="switchLanguage('chinese')">中文</button>
  <button onclick="switchLanguage('japanese')">日本語</button>
</div>

<!-- Audio Player -->
<audio id="multilingual-audio" controls>
  <source src="/assets/audio/homepage-english.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

<script>
function switchLanguage(language) {
  var audioPlayer = document.getElementById('multilingual-audio');
  var basePath = '/assets/audio/homepage-';
  
  switch(language) {
    case 'english':
      audioPlayer.src = basePath + 'english.wav';
      break;
    case 'spanish':
      audioPlayer.src = basePath + 'spanish.wav';
      break;
    case 'chinese':
      audioPlayer.src = basePath + 'chinese.wav';
      break;
    case 'japanese':
      audioPlayer.src = basePath + 'japanese.wav';
      break;
    default:
      // Default to English if something goes wrong
      audioPlayer.src = basePath + 'english.wav';
  }
  
  // Reload and play the new audio file
  audioPlayer.load();
  audioPlayer.play();
}
</script>

<style>
  #audio-controls button {
  cursor: pointer;
  padding: 10px 15px;
  margin: 0 10px;
  background-color: #0c041c; /* A bright, modern blue */
  border: none;
  border-radius: 5px;
  color: white; /* White text on a blue background */
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  text-transform: uppercase; /* Makes the button text uppercase */
  letter-spacing: 1px; /* Increases spacing between letters for a modern look */
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); /* Adds a subtle shadow for depth */
  transition: background-color 0.3s, box-shadow 0.3s; /* Smooth transitions for hover effects */
}

#audio-controls button:hover, #audio-controls button:focus {
  background-color: #d2a2ea; /* A darker blue on hover/focus for interaction feedback */
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); /* A larger shadow on hover/focus for a "lifting" effect */
  outline: none; /* Removes the outline to keep the design clean */
}

#audio-controls button:active {
  background-color: #69588c; /* Even darker for the active (clicked) state */
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); /* Reverts to a smaller shadow to simulate pressing down */
}
</style>



I am **Qi Chen**, a passionate algorithm engineer with a love for turning complex datasets into actionable insights. With a background in applied physics and data science, I bring a rigorous analytical approach to solving problems.

As a machine learning engineer with a passion for leveraging cutting-edge technologies to solve complex problems, I take pride in my ability to drive innovation and deliver impactful solutions. With a strong foundation in data science, mathematics, and computer science, I possess a comprehensive understanding of machine learning algorithms, deep learning architectures, and their practical applications.

### What I Do
- Data Analysis
- Machine Learning
- Data-Driven Decision Making


Explore my website to learn more about my professional journey, key projects, and personal insights I've gathered along the way. 


Let's connect and create something impactful together!

I'm always open to discussing new technologies, project ideas, or potential collaborations. 
 
