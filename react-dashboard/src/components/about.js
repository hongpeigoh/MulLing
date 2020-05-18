import React, { Component } from 'react';
import Loading from './widget.js'

import marked from "marked";

class AboutHead extends Component {
  render() {
    return(
      <div className="app-header" id="about-header">
        <div className="app-container container">
          <div className="row textcontent center">
            <h1>What is MulLing?</h1>
            <h4>MulLing stands for <strong>Mul</strong>ti-<strong>ling</strong>ual Information Retrieval, a project exploring state-of-the-art techniques in translating for an information retrieval tasks using methods in machine learning.</h4>
          </div>
        </div>
      </div>
    )
  }
}

class About extends Component {
  constructor(props) {
    super(props);
    this.state = { isLoading: true }
  }
  
  componentDidMount() {
    document.getElementById('app-container').style.transform = "none";
    document.getElementById('app-container').style.maxWidth= "1140px";
    const readmePath = require("%PUBLIC_URL%/../../../README.md");
  
    fetch(readmePath)
      .then(response => {
        return response.text()
      })
      .then(text => {
        text = text.replace('./dump/results2.png', 'https://lh3.googleusercontent.com/KaEFK3zTVI5t3twtAoOHHhRKvY_EuirsPpSH-JDNFTLhuMKmQqQQbYrzIU4gfM_K7oZOm3SO6DpErzFsMh5mkfe7LY9Lqa5yB8R7bJKV=s1167')
        text = text.replace('./dump/results1.png', 'https://lh5.googleusercontent.com/cIUlgkHkvpkvZmOpwmjETqqtTb_OHnfq_H9REXhlFi61qDquqt1mDROr6BUz-IWAaOEoq69o08FAMzOiba1Oo_AR5zy6eQzgUCqHZ3tc=s1167')
        this.setState({
          markdown: marked(text)
        })
      })
    document.getElementById('loading').style.display = "none";
  }

  render() {
    const { markdown } = this.state;

    return (
      <div className='app-container container about width-70' id='app-container'>
        <div className='row textcontent'>
          <Loading/>
          <article dangerouslySetInnerHTML={{__html: markdown}} onLoad={(e)=> document.getElementById('loading2').style.display = "none"}/>
        </div>
      </div>
    )
  }
}



export { About, AboutHead };