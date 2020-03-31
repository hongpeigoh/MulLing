import React, { Component } from 'react';
import Loading from './widget.js'

import marked from "marked";

class About extends Component {
  constructor(props) {
    super(props);
    this.state = { isLoading: true }
  }
  
  componentDidMount() {
    document.getElementById('app-container').style.transform = "none";
    document.getElementById('app-container').style.maxWidth= "1140px";
    const readmePath = require("../images/README.md");
  
    fetch(readmePath)
      .then(response => {
        return response.text()
      })
      .then(text => {
        this.setState({
          markdown: marked(text)
        })
      })
    document.getElementById('loading').style.display = "none";
  }

  render() {
    const { markdown } = this.state;

    return (
      <div className='about'>
        <div className="row block"/>
        <h1>About</h1>
        <div className='row textcontent'>
          <Loading/>
          <article dangerouslySetInnerHTML={{__html: markdown}} onLoad={(e)=> document.getElementById('loading2').style.display = "none"}/>
        </div>
      </div>
    )
  }
}



export default About