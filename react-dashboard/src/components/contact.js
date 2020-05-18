import React, { Component } from 'react';
import Loading from './widget.js'
import GitHub from "../images/github_black.svg";
import Facebook from "../images/facebook_black.svg";
import Email from "../images/email_black.svg";
import { Button } from '@progress/kendo-react-buttons';

class ContactHead extends Component {
  render() {
    return(
      <div className="app-header" id="contact-header">
        <div className="app-container container">
          <div className="row textcontent center">
            <h1>Find Us</h1>
            <h4>Have a suggestion? Reporting a bug? Fill in the feedback form below!</h4>
          </div>
        </div>
      </div>
    )
  }
}


class Contact extends Component {
  constructor(props) {
    super(props);
    this.state = { isLoading: true }
  }
  componentDidMount(){
    document.getElementById('app-container').style.transform = "none";
    document.getElementById('app-container').style.maxWidth= "1140px";
  }
  btnClick(clickedurl) {
    window.open(clickedurl);
  }
  render() {
    return(
      <div className='app-container container contact width-70' id='app-container'>
        <div className="row textcontent">
          <div className="col-4">
            <div className="clickicon center">
              <img src= {GitHub} alt="GitHub"/>
              <h2>GitHub</h2>
              <p>Access the source code repository and more.</p>
              <Button primary={true} icon="redo" onClick={this.btnClick.bind(this, "http://www.github.com/hongpeigoh/MulLing")}>
                Link
              </Button>
            </div>
          </div>
          <div className="col-4">
            <div className="clickicon center">
              <img src= {Facebook} alt="Facebook"/>
              <h2>Facebook</h2>
              <p>Find me on social media.</p>
              <Button primary={true} icon="redo" onClick={this.btnClick.bind(this, "http://facebook.com/hongpeigoh")}>
                Link
              </Button>
            </div>
          </div>
          <div className="col-4">
            <div className="clickicon center">
              <img src= {Email} alt="Email"/>
              <h2>E-mail</h2>
              <p>Drop me an e-mail.</p>
            <Button primary={true} icon="redo" onClick={this.btnClick.bind(this, "mailto:hong_pei_99@hotmail.com")}>
              Link
            </Button>
            </div>
          </div>
        </div>
        <div className="row textcontent center">
          <Loading/>
          <iframe title="feedback" src="http://localhost:5050/feedback"
                    width="500"
                    height="700"
                    frameBorder="0"
                    marginHeight="0"
                    marginWidth="0"
                    style={{overflow:"hidden"}}
                    onLoad={(e)=> document.getElementById('loading').style.display = "none"}>Loading…</iframe>
        </div>
      </div>
    )
  }
}

export { Contact, ContactHead };