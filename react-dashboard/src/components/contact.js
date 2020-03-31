import React, { Component } from 'react';
import Loading from './widget.js'

class Contact extends Component {
  constructor(props) {
    super(props);
    this.state = { isLoading: true }
  }
  componentDidMount(){
    document.getElementById('app-container').style.transform = "none";
    document.getElementById('app-container').style.maxWidth= "1140px";
  }
  render() {
    return(
      <div className='contact'>
        <div className="row block"/>
        <h1>Contact</h1>
        <div className="row">
          <div className="col-6">
          <Loading/>
          <iframe title="feedback" src="http://192.168.137.217:5050/feedback"
                    width="500"
                    height="700"
                    frameBorder="0"
                    marginHeight="0"
                    marginWidth="0"
                    style={{overflow:"hidden"}}
                    onLoad={(e)=> document.getElementById('loading').style.display = "none"}>Loading…</iframe>
          </div>
          <div className="col-6 textcontent" style={{margin: "1vh 0"}}>
            <div className="row">
              <article style={{padding: "10px"}}>
                <h2>Creator: Goh Hong Pei</h2>
                <ul>
                  <li><strong>Github:</strong> github.com/hongpeigoh</li>
                  <li><strong>E-mail:</strong> hong_pei_99@hotmail.com</li>
                  <li><strong>Facebook:</strong> fb.com/hongpeigoh</li>
                </ul>
                <h2>Mentor: Ivan</h2>
                <h2>Technical Mentor: Avery Khoo</h2>
              </article>
            </div>
          </div>
        </div>
      </div>
    )
  }
}

export default Contact