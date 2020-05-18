
import React from 'react';
const isPresent = value => value !== null && value !== undefined;

export default function withValueField(DropDownComponent) {
    return class extends React.Component {
        component;
        events = {
            onBlur: event => this.triggerEvent('onBlur', event),
            onFocus: event => this.triggerEvent('onFocus', event),
            onChange: event => this.triggerEvent('onChange', event),
            onPageChange: event => this.triggerEvent('onPageChange', event),
            onFilterChange: event => this.triggerEvent('onFilterChange', event)
        };

        get value() {
            if (this.component) {
                const value = this.component.value;
                return isPresent(value) ? value[this.props.valueField] : value;
            }
        }

        render() {
            return (
                <DropDownComponent
                    {...this.props}
                    value={this.itemFromValue(this.props.value)}
                    defaultValue={this.itemFromValue(this.props.defaultValue)}
                    ref={component => this.component = component}
                    {...this.events}
                />
            );
        }

        triggerEvent(eventType, event) {
            if (this.props[eventType]) {
                this.props[eventType].call(undefined, {
                    ...event,
                    target: this
                });
            }
        }

        itemFromValue(value) {
            const { data = [], valueField } = this.props;
            return isPresent(value) ?
                data.find(item => item[valueField] === value) : value;
        }
    };
}
