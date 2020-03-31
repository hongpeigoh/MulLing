import React, { Component } from 'react';
import { Button } from '@progress/kendo-react-buttons';
import GitHub from "../images/github.svg";
import { NavLink } from 'react-router-dom';

class Footer extends Component{
  btnClick(clickedurl) {
    window.open(clickedurl);
  }
  render() {
    return(
      <footer className="site-footer">
      <div className="container">
        <div className="row">
          <div className="col-4 footer-groups">
            <h2>Site map</h2>
            <div>
              <nav>
                <NavLink to='/'><Button look="flat" icon="layout">Home</Button></NavLink>
                <NavLink to='/about'><Button look="flat" icon="info">About</Button></NavLink>
                <NavLink to='/contact'><Button look="flat" icon="inbox">Contact</Button></NavLink>
              </nav>
            </div>
          </div>
            
          <div className="col-4 footer-groups">
            <h2>Built by</h2>
            <div><small>
                © 2020 Goh Hong Pei
              </small>
            </div>
          </div>

          <div className="col-4 footer-groups">
            <h2>Find Us</h2>
            <div>
                <Button look="flat" onClick={this.btnClick.bind(this, "http://www.github.com/hongpeigoh/MulLing")}>
                  <img src= {GitHub} width="16px" alt="GitHub"/>
                    GitHub
                </Button>   
                <Button look="flat" onClick={this.btnClick.bind(this, "mailto:hong_pei_99@hotmail.com")}>
                  <span className="k-icon k-i-email" />
                    Email
                </Button>
                <Button look="flat" onClick={this.btnClick.bind(this, "http://facebook.com/hongpeigoh")}>
                  <span className="k-icon k-i-facebook" />
                    Facebook
                </Button>
            </div>
          </div>

        </div>
      </div>
    </footer>
    )
  }
}

export default Footer