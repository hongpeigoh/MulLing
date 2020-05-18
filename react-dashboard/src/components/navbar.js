import React, { Component } from 'react';
import { Button } from '@progress/kendo-react-buttons';
import { NavLink } from 'react-router-dom';
import Icon from '../images/logo256.png'

class Header extends Component{
  render() {
    return(
      <footer className="site-header">
      <div className="container">
        <div className="row">
          <div className="col-6 col-md-4 col-xl-3 icon-holder">
            <img src={ Icon } width="48px" height="48px" alt="MulLing"></img>
            <h1>MulLing</h1>
            <small>
                © 2020 Goh Hong Pei
            </small>
          </div>
          <div className="col-6 col-md-8 col-xl-9">
              <nav>
                <NavLink to='/'><Button look="flat" icon="layout">Home</Button></NavLink>
                <NavLink to='/about'><Button look="flat" icon="info">About</Button></NavLink>
                <NavLink to='/contact'><Button look="flat" icon="inbox">Contact</Button></NavLink>
                <NavLink to='/sandbox'><Button look="flat" icon="apply-format">Sandbox</Button></NavLink>
              </nav>
          </div>
        </div>
      </div>
    </footer>
    )
  }
}

export default Header