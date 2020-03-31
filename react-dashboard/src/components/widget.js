import React from 'react';
import { Button } from '@progress/kendo-react-buttons';

const Loading = () => (
    <span className="center">
      <Button id="loading" disabled={true} icon="loading">Loading...</Button>
    </span>
)

export default Loading