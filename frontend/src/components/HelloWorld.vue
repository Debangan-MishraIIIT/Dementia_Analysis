<template>
  <div class="container">
    <h1>Image Upload</h1>
    <input type="file" @change="onFileChange" />
    <button @click="uploadImage">Upload Image</button>
    <div v-if="response">
      <h2>Response:</h2>
      <pre>{{ response }}</pre>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      file: null,
      response: null,
    };
  },
  methods: {
    onFileChange(event) {
      this.file = event.target.files[0];
    },
    async uploadImage() {
      if (!this.file) {
        alert('Please select a file first.');
        return;
      }

      const formData = new FormData();
      formData.append('file', this.file);

      try {
        const res = await axios.post('http://localhost:5000/image', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        this.response = res.data;
      } catch (error) {
        this.response = error.response.data;
      }
    },
  },
};
</script>

<style scoped>
.container {
  max-width: 600px;
  margin: 0 auto;
  text-align: center;
}
</style>
