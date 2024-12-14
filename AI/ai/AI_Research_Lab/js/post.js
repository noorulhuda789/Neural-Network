
const posts = {
  post1: {
    title: "Exploring the Great Outdoors",
    meta: "Travel . Adventure | June 18, 2024 | by Alex Johnson",
    content:
      "<p>Discover the beauty and adventure of exploring nature. From hiking trails to scenic views, find out why the great outdoors is a perfect getaway. Whether you're a seasoned explorer or a weekend warrior, there's something for everyone to enjoy in the wilderness. So pack your bags, lace up your boots, and get ready to embark on a journey like no other.</p>",
  },
  post2: {
    title: "The Future of Technology",
    meta: "Tech . Innovation | June 15, 2024 | by Maria Lee",
    content:
      "<p>Dive into the latest advancements in technology. From AI to quantum computing, explore what the future holds for tech enthusiasts. The world of technology is ever-evolving, with new innovations being introduced at a rapid pace. Stay ahead of the curve by understanding the trends and breakthroughs that are shaping our future.</p>",
  },
  post3: {
    title: "Healthy Living Tips",
    meta: "Health . Wellness | June 10, 2024 | by John Smith",
    content:
      "<p>Learn the secrets to maintaining a healthy lifestyle. From diet and exercise to mental well-being, find tips that will help you live a balanced life. Health is more than just the absence of illness; it's about thriving in all aspects of your life. Discover the best practices for staying healthy and happy, both physically and mentally.</p>",
  },
  post4: {
    title: "Mastering Photography",
    meta: "Art . Photography | June 5, 2024 | by Sarah Brown",
    content:
      "<p>Enhance your photography skills with these expert tips. From composition to lighting, learn how to capture stunning photos every time. Photography is both an art and a science, requiring a keen eye and a good understanding of technical elements. Unlock your potential as a photographer with these practical and creative tips.</p>",
  },
  post5: {
    title: "Culinary Adventures",
    meta: "Food . Travel | May 31, 2024 | by Chef Mark",
    content:
      "<p>Embark on a culinary journey with our guide to exotic recipes and cooking techniques. Discover new flavors and cooking styles from around the world. Food is a universal language that brings people together. Explore the diverse culinary traditions of different cultures and bring a taste of the world to your kitchen.</p>",
  },
  post6: {
    title: "The Art of Minimalism",
    meta: "Lifestyle . Design | June 1, 2024 | by Emily White",
    content:
      "<p>Discover the benefits of minimalism and how it can transform your life. Learn tips on decluttering, simplifying, and creating a minimalist lifestyle. Minimalism is about more than just getting rid of excess; it's about making room for what truly matters. Find out how to live a more intentional and fulfilling life by embracing minimalism.</p>",
  },
};

function getPostId() {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get("post");
}

function loadPost() {
  const postId = getPostId();
  if (posts[postId]) {
    document.getElementById("post-title").innerText = posts[postId].title;
    document.getElementById("post-meta").innerText = posts[postId].meta;
    document.getElementById("post-content").innerHTML =
      posts[postId].content;
  } else {
    document.getElementById("post-title").innerText = "Post Not Found";
    document.getElementById("post-meta").innerText = "";
    document.getElementById("post-content").innerHTML =
      "<p>The post you are looking for does not exist.</p>";
  }
}

window.onload = loadPost;