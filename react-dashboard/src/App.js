import React, { Component } from 'react';
import { Switch, Route, BrowserRouter } from 'react-router-dom';
import { Ripple } from '@progress/kendo-react-ripple';

import About from './components/about'
import Home from './components/home'
import Contact from'./components/contact'
import Footer from './components/navbar'

import '@progress/kendo-theme-material/dist/all.css';
import 'bootstrap-4-grid/css/grid.min.css';
import './App.css';

class App extends Component {
  constructor(props) {
    super(props);
    this.state = { isLoading: true }
  }

  render() {
    return (
        <Ripple><BrowserRouter>
          <div className="bootstrap-wrapper">
            <div className="app-container container" id="app-container">
              <Main />
            </div>
          </div>
          <Footer />
        </BrowserRouter></Ripple>
        
    )
  }
}

const Main = () => (
  <Switch>
    <Route exact path='/' component={Home}></Route>
    <Route exact path='/about' component={About}></Route>
    <Route exact path='/contact' component={Contact}></Route>
  </Switch>
);

export default App;